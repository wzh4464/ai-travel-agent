"""Tests for the transit-hub blacklist filter on canonical Flight dicts."""

from __future__ import annotations

from agents.presentation.sorting import _touches_banned_transit, filter_flights


def _flight(legs: list[tuple[str, str]], *, price: float = 100.0) -> dict:
    """Build a minimal Flight dict from a list of (departure, arrival) legs."""
    return {
        'flight_id': 'f',
        'price': price,
        'currency': 'USD',
        'total_duration_minutes': 600,
        'stops': max(0, len(legs) - 1),
        'legs': [
            {
                'departure_airport': dep,
                'departure_time': '',
                'arrival_airport': arr,
                'arrival_time': '',
                'airline': 'XX',
                'flight_number': 'XX1',
                'duration_minutes': 300,
                'aircraft': '',
                'cabin_class': 'economy',
            }
            for dep, arr in legs
        ],
        'provider': 'stub',
    }


BANNED_ME = {'DXB', 'DOH', 'AUH', 'RUH'}


class TestTouchesBannedTransit:
    def test_non_stop_is_never_a_transit(self):
        # Even if the destination matches the banned set, a non-stop is fine.
        flight = _flight([('HKG', 'DXB')])
        assert _touches_banned_transit(flight, BANNED_ME) is False

    def test_one_stop_via_dxb_blocked(self):
        flight = _flight([('HKG', 'DXB'), ('DXB', 'CDG')])
        assert _touches_banned_transit(flight, BANNED_ME) is True

    def test_one_stop_via_ams_allowed(self):
        flight = _flight([('HKG', 'AMS'), ('AMS', 'CDG')])
        assert _touches_banned_transit(flight, BANNED_ME) is False

    def test_two_stops_one_bad(self):
        flight = _flight([('HKG', 'AMS'), ('AMS', 'DOH'), ('DOH', 'CDG')])
        assert _touches_banned_transit(flight, BANNED_ME) is True

    def test_destination_equal_to_banned_is_not_a_transit(self):
        """A user who *wants* to fly to DXB should not be filtered."""
        flight = _flight([('HKG', 'BKK'), ('BKK', 'DXB')])
        # The final arrival airport (DXB) is the destination, not a transit
        assert _touches_banned_transit(flight, BANNED_ME) is False

    def test_empty_banned_set_never_blocks(self):
        flight = _flight([('HKG', 'DXB'), ('DXB', 'CDG')])
        assert _touches_banned_transit(flight, set()) is False


class TestFilterFlightsWithAvoidTransit:
    def test_filter_removes_banned_transit(self):
        flights = [
            _flight([('HKG', 'DXB'), ('DXB', 'CDG')], price=4800),  # should go
            _flight([('HKG', 'AMS'), ('AMS', 'CDG')], price=5600),  # kept
            _flight([('HKG', 'CDG')], price=6800),                    # kept
        ]
        kept = filter_flights(flights, avoid_transit=BANNED_ME)
        assert len(kept) == 2
        assert all(not _touches_banned_transit(f, BANNED_ME) for f in kept)

    def test_combines_with_other_filters(self):
        flights = [
            _flight([('HKG', 'AMS'), ('AMS', 'CDG')], price=5600),
            _flight([('HKG', 'CDG')], price=6800),
            _flight([('HKG', 'DXB'), ('DXB', 'CDG')], price=4800),
        ]
        kept = filter_flights(flights, avoid_transit=BANNED_ME, max_price=6000)
        assert len(kept) == 1
        assert kept[0]['price'] == 5600
