"""Contract tests for the per-provider Flight normalizers.

These tests pin the mapping from each provider's raw payload shape to the
canonical :class:`agents.data_sources.normalizer.Flight` schema. If an
upstream API changes a field, the fixture JSON can be updated and the
assertion diff will tell you exactly which bit drifted.
"""

from __future__ import annotations

from agents.data_sources.normalizer import (
    _minutes_between,
    _parse_iso_duration,
    normalize_amadeus,
    normalize_duffel,
    normalize_kiwi,
    normalize_serpapi,
)


class TestISOHelpers:
    def test_parse_iso_duration_hours_and_minutes(self):
        assert _parse_iso_duration('PT11H30M') == 11 * 60 + 30

    def test_parse_iso_duration_only_hours(self):
        assert _parse_iso_duration('PT2H') == 120

    def test_parse_iso_duration_only_minutes(self):
        assert _parse_iso_duration('PT45M') == 45

    def test_parse_iso_duration_empty(self):
        assert _parse_iso_duration('') == 0

    def test_parse_iso_duration_garbage(self):
        assert _parse_iso_duration('not-a-duration') == 0

    def test_minutes_between_simple(self):
        assert _minutes_between('2026-05-01T08:00:00', '2026-05-01T11:30:00') == 210

    def test_minutes_between_z_suffix(self):
        assert _minutes_between('2026-05-01T08:00:00Z', '2026-05-01T11:30:00Z') == 210

    def test_minutes_between_invalid_returns_zero(self):
        assert _minutes_between('bogus', '2026-05-01T11:30:00') == 0

    def test_minutes_between_end_before_start_clamps_to_zero(self):
        assert _minutes_between('2026-05-01T12:00:00', '2026-05-01T11:00:00') == 0


class TestSerpAPINormalizer:
    def test_non_stop_flight_round_trip(self, serpapi_flights_fixture):
        raw = serpapi_flights_fixture['best_flights'][0]
        flight = normalize_serpapi(raw).to_dict()

        assert flight['price'] == 520.0
        assert flight['currency'] == 'USD'
        assert flight['stops'] == 0
        assert flight['total_duration_minutes'] == 690
        assert flight['provider'] == 'serpapi-google-flights'
        assert flight['airline_logo'].endswith('AA.png')
        assert len(flight['legs']) == 1
        leg = flight['legs'][0]
        assert leg['airline'] == 'American Airlines'
        assert leg['departure_airport'] == 'JFK'
        assert leg['arrival_airport'] == 'LHR'
        assert leg['cabin_class'] == 'Economy'

    def test_one_stop_flight(self, serpapi_flights_fixture):
        raw = serpapi_flights_fixture['best_flights'][1]
        flight = normalize_serpapi(raw).to_dict()

        assert flight['stops'] == 1
        assert len(flight['legs']) == 2
        assert flight['legs'][0]['departure_airport'] == 'JFK'
        assert flight['legs'][-1]['arrival_airport'] == 'LHR'

    def test_stable_flight_id_is_deterministic(self, serpapi_flights_fixture):
        raw = serpapi_flights_fixture['best_flights'][0]
        a = normalize_serpapi(raw).to_dict()['flight_id']
        b = normalize_serpapi(raw).to_dict()['flight_id']
        assert a == b


class TestAmadeusNormalizer:
    def test_non_stop_offer(self, amadeus_offers_fixture):
        carriers = amadeus_offers_fixture['dictionaries']['carriers']
        offer = amadeus_offers_fixture['data'][0]
        flight = normalize_amadeus(offer, carriers).to_dict()

        assert flight['provider'] == 'amadeus'
        assert flight['flight_id'] == '1'
        assert flight['price'] == 420.5
        assert flight['currency'] == 'USD'
        assert flight['total_duration_minutes'] == 11 * 60 + 30
        assert flight['stops'] == 0

        leg = flight['legs'][0]
        assert leg['airline'] == 'AMERICAN AIRLINES'
        assert leg['flight_number'] == 'AA100'
        assert leg['departure_airport'] == 'JFK'
        assert leg['arrival_airport'] == 'LHR'
        assert leg['cabin_class'] == 'economy'
        assert leg['aircraft'] == '777'

    def test_multi_segment_offer(self, amadeus_offers_fixture):
        carriers = amadeus_offers_fixture['dictionaries']['carriers']
        offer = amadeus_offers_fixture['data'][1]
        flight = normalize_amadeus(offer, carriers).to_dict()

        assert flight['flight_id'] == '2'
        assert flight['stops'] == 1
        assert flight['total_duration_minutes'] == 13 * 60 + 45
        assert [leg['departure_airport'] for leg in flight['legs']] == ['JFK', 'CDG']
        assert [leg['arrival_airport'] for leg in flight['legs']] == ['CDG', 'LHR']
        assert all(leg['cabin_class'] == 'economy' for leg in flight['legs'])

    def test_missing_carriers_dict_falls_back_to_code(self, amadeus_offers_fixture):
        offer = amadeus_offers_fixture['data'][0]
        flight = normalize_amadeus(offer, {}).to_dict()
        assert flight['legs'][0]['airline'] == 'AA'


class TestKiwiNormalizer:
    def test_non_stop_offer(self, kiwi_search_fixture):
        offer = kiwi_search_fixture['data'][0]
        flight = normalize_kiwi(offer).to_dict()

        assert flight['provider'] == 'kiwi'
        assert flight['flight_id'] == 'kw-offer-1'
        assert flight['price'] == 399.0
        assert flight['currency'] == 'USD'
        assert flight['total_duration_minutes'] == 41400 // 60
        assert flight['stops'] == 0
        assert flight['booking_url'] == 'https://www.kiwi.com/book/kw-offer-1'

        leg = flight['legs'][0]
        assert leg['airline'] == 'BA'
        assert leg['flight_number'] == 'BA178'
        assert leg['departure_airport'] == 'JFK'
        assert leg['arrival_airport'] == 'LHR'

    def test_one_stop_offer(self, kiwi_search_fixture):
        offer = kiwi_search_fixture['data'][1]
        flight = normalize_kiwi(offer).to_dict()

        assert flight['stops'] == 1
        assert [leg['airline'] for leg in flight['legs']] == ['KL', 'KL']
        assert flight['legs'][0]['flight_number'] == 'KL644'
        assert flight['legs'][1]['flight_number'] == 'KL1013'

    def test_missing_duration_defaults_to_zero(self):
        offer = {
            'id': 'kw-bad',
            'price': 100,
            'route': [{
                'airline': 'XX', 'flight_no': 1,
                'flyFrom': 'AAA', 'flyTo': 'BBB',
                'local_departure': '', 'local_arrival': '',
            }],
        }
        flight = normalize_kiwi(offer).to_dict()
        assert flight['total_duration_minutes'] == 0


class TestDuffelNormalizer:
    def test_non_stop_offer(self, duffel_offer_request_fixture):
        offer = duffel_offer_request_fixture['data']['offers'][0]
        flight = normalize_duffel(offer).to_dict()

        assert flight['provider'] == 'duffel'
        assert flight['flight_id'] == 'off_0000AaaaNonstop'
        assert flight['price'] == 5480.0
        assert flight['currency'] == 'HKD'
        assert flight['total_duration_minutes'] == 13 * 60
        assert flight['stops'] == 0

        leg = flight['legs'][0]
        assert leg['airline'] == 'Cathay Pacific'
        assert leg['flight_number'] == 'CX261'
        assert leg['departure_airport'] == 'HKG'
        assert leg['arrival_airport'] == 'CDG'
        assert leg['cabin_class'] == 'economy'
        assert leg['aircraft'] == '77W'

    def test_one_stop_via_dxb(self, duffel_offer_request_fixture):
        offer = duffel_offer_request_fixture['data']['offers'][1]
        flight = normalize_duffel(offer).to_dict()

        assert flight['flight_id'] == 'off_0000AbbbVaDXB'
        assert flight['price'] == 4820.0
        assert flight['stops'] == 1
        assert len(flight['legs']) == 2
        # Intermediate stop is DXB
        assert flight['legs'][0]['arrival_airport'] == 'DXB'
        assert flight['legs'][1]['departure_airport'] == 'DXB'

    def test_multi_slice_round_trip_reports_worst_slice_stops(self):
        """A round-trip with a non-stop return and a 1-stop outbound should
        report ``stops = 1`` (the worst leg), not 0 and not 2."""
        offer = {
            'id': 'rt',
            'total_amount': '1000.00',
            'total_currency': 'USD',
            'slices': [
                {
                    'duration': 'PT5H',
                    'segments': [
                        {
                            'origin': {'iata_code': 'A'},
                            'destination': {'iata_code': 'B'},
                            'departing_at': '', 'arriving_at': '', 'duration': 'PT2H30M',
                            'marketing_carrier': {'iata_code': 'XX'},
                            'marketing_carrier_flight_number': '1',
                            'passengers': [{'cabin_class': 'economy'}],
                        },
                        {
                            'origin': {'iata_code': 'B'},
                            'destination': {'iata_code': 'C'},
                            'departing_at': '', 'arriving_at': '', 'duration': 'PT2H30M',
                            'marketing_carrier': {'iata_code': 'XX'},
                            'marketing_carrier_flight_number': '2',
                            'passengers': [{'cabin_class': 'economy'}],
                        },
                    ],
                },
                {
                    'duration': 'PT5H',
                    'segments': [
                        {
                            'origin': {'iata_code': 'C'},
                            'destination': {'iata_code': 'A'},
                            'departing_at': '', 'arriving_at': '', 'duration': 'PT5H',
                            'marketing_carrier': {'iata_code': 'XX'},
                            'marketing_carrier_flight_number': '3',
                            'passengers': [{'cabin_class': 'economy'}],
                        },
                    ],
                },
            ],
        }
        flight = normalize_duffel(offer).to_dict()
        assert flight['stops'] == 1
        assert flight['total_duration_minutes'] == 10 * 60

    def test_missing_marketing_carrier_name_falls_back_to_code(self):
        offer = {
            'id': 'x', 'total_amount': '0', 'total_currency': 'USD',
            'slices': [{
                'duration': 'PT1H',
                'segments': [{
                    'origin': {'iata_code': 'A'}, 'destination': {'iata_code': 'B'},
                    'departing_at': '', 'arriving_at': '', 'duration': 'PT1H',
                    'marketing_carrier': {'iata_code': 'ZZ'},  # no 'name'
                    'marketing_carrier_flight_number': '99',
                    'passengers': [],
                }],
            }],
        }
        flight = normalize_duffel(offer).to_dict()
        assert flight['legs'][0]['airline'] == 'ZZ'
