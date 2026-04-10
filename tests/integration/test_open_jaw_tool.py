"""Integration tests for the open_jaw_search tool.

Stubs the aggregator with a canned source so the tool runs end-to-end
(fan-out, ranking, formatting) without hitting any upstream provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.data_sources.aggregator import AggregatedFlightSource
from agents.data_sources.base import BaseFlightSource
from agents.errors import NoResultsError
from agents.tools.open_jaw import OpenJawInput, open_jaw_search


def _flight(fid: str, origin: str, dest: str, price: float, *,
            via: list[str] | None = None, currency: str = 'HKD') -> dict:
    via = via or []
    legs_airports = [origin, *via, dest]
    legs = [
        {
            'departure_airport': legs_airports[i],
            'departure_time': '',
            'arrival_airport': legs_airports[i + 1],
            'arrival_time': '',
            'airline': 'Test Air',
            'flight_number': 'TA1',
            'duration_minutes': 300,
            'aircraft': '',
            'cabin_class': 'economy',
        }
        for i in range(len(legs_airports) - 1)
    ]
    return {
        'flight_id': fid,
        'price': price,
        'currency': currency,
        'total_duration_minutes': 300 * len(legs),
        'stops': max(0, len(legs) - 1),
        'legs': legs,
        'provider': 'stub',
    }


class _CannedSource(BaseFlightSource):
    """Returns scripted flights based on an (origin, destination) table."""

    name = 'canned'

    def __init__(self, table: dict[tuple[str, str], list[dict]]):
        self.name = 'canned'
        self._table = table
        self.calls: list[tuple[str, str]] = []

    def is_configured(self) -> bool:
        return True

    def search(self, *, origin, destination, outbound_date, **kwargs):
        self.calls.append((origin, destination))
        flights = self._table.get((origin, destination))
        if flights is None:
            raise NoResultsError(origin, destination, outbound_date)
        return list(flights)

    def details(self, flight_id):
        return {'status': 'unsupported'}


@pytest.fixture
def canned_aggregator(monkeypatch):
    """Build a canned table that simulates a tiny Europe-from-HKG market."""
    table: dict[tuple[str, str], list[dict]] = {
        # Outbound: HKG -> Europe
        ('HKG', 'CDG'): [
            _flight('hkg-cdg-direct', 'HKG', 'CDG', 5400),
            _flight('hkg-cdg-dxb', 'HKG', 'CDG', 4200, via=['DXB']),
        ],
        ('HKG', 'FCO'): [
            _flight('hkg-fco-doh', 'HKG', 'FCO', 4100, via=['DOH']),
            _flight('hkg-fco-ams', 'HKG', 'FCO', 5100, via=['AMS']),
        ],
        ('HKG', 'LHR'): [
            _flight('hkg-lhr-direct', 'HKG', 'LHR', 5700),
        ],
        ('HKG', 'MAD'): [],  # No flights on this route
        ('HKG', 'AMS'): [
            _flight('hkg-ams-direct', 'HKG', 'AMS', 5200),
        ],
        # Return: Europe -> HKG
        ('CDG', 'HKG'): [
            _flight('cdg-hkg-direct', 'CDG', 'HKG', 5500),
        ],
        ('FCO', 'HKG'): [
            _flight('fco-hkg-doh', 'FCO', 'HKG', 4300, via=['DOH']),
            _flight('fco-hkg-ist', 'FCO', 'HKG', 4800, via=['IST']),
        ],
        ('LHR', 'HKG'): [
            _flight('lhr-hkg-direct', 'LHR', 'HKG', 5600),
        ],
        ('MAD', 'HKG'): [],
        ('AMS', 'HKG'): [
            _flight('ams-hkg-direct', 'AMS', 'HKG', 5300),
        ],
    }
    source = _CannedSource(table)
    agg = AggregatedFlightSource([source])
    # Replace the default aggregator with our canned one.
    import agents.tools.open_jaw as ojs_mod
    monkeypatch.setattr(ojs_mod, 'get_default_aggregator', lambda: agg)
    return agg, source


def _run(params: dict[str, Any]) -> dict:
    return open_jaw_search.func(OpenJawInput(**params))


class TestOpenJawOrchestration:
    def test_hong_kong_to_europe_cheap(self, canned_aggregator):
        """The canonical user query: "从香港去欧洲 4/23-5/3 便宜 不要中东中转"."""
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
            'top_n': 5,
        })
        assert result['status'] == 'ok'
        assert result['origin'] == 'HKG'
        assert 'CDG' in result['candidates']

        # Banned transit list was expanded to the Middle East hubs
        assert 'DXB' in result['banned_transit']
        assert 'DOH' in result['banned_transit']

        # Every surviving itinerary must avoid Middle East transits
        for combo in result['combinations']:
            for flight in (combo['outbound'], combo['return']):
                mids = [leg['arrival_airport'] for leg in flight['legs'][:-1]]
                assert 'DXB' not in mids
                assert 'DOH' not in mids

    def test_results_are_sorted_by_total_price(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
        })
        prices = [c['total_price'] for c in result['combinations']]
        assert prices == sorted(prices)

    def test_open_jaw_is_allowed_by_default(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
        })
        # There should be at least one combination where entry != exit
        assert any(c['open_jaw'] for c in result['combinations'])

    def test_same_city_constraint_honoured(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
            'same_city': True,
        })
        assert all(not c['open_jaw'] for c in result['combinations'])
        assert all(c['entry_city'] == c['exit_city'] for c in result['combinations'])

    def test_middle_east_strict_also_blocks_istanbul(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east_strict'],
        })
        assert 'IST' in result['banned_transit']
        for combo in result['combinations']:
            for flight in (combo['outbound'], combo['return']):
                mids = [leg['arrival_airport'] for leg in flight['legs'][:-1]]
                assert 'IST' not in mids

    def test_fan_out_called_for_every_candidate(self, canned_aggregator):
        _agg, source = canned_aggregator
        _run({
            'origin': 'HKG',
            # Explicit comma list so we control the pairs exactly
            'destination_region': 'CDG,FCO,LHR',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
        })
        # Outbound + return = 2 calls per candidate
        wanted_pairs = {
            ('HKG', 'CDG'), ('HKG', 'FCO'), ('HKG', 'LHR'),
            ('CDG', 'HKG'), ('FCO', 'HKG'), ('LHR', 'HKG'),
        }
        actual = set(source.calls)
        assert wanted_pairs.issubset(actual)

    def test_individual_route_failures_do_not_abort_batch(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',  # includes MAD which has no flights
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
        })
        # MAD raises NoResultsError inside the aggregator but the tool
        # should still produce combinations for the other cities.
        assert result['status'] == 'ok'
        assert result['count'] >= 1

    def test_max_price_cap(self, canned_aggregator):
        # Cheapest same-city combos after middle_east filter:
        #   AMS rt = 5200+5300 = 10500
        #   CDG rt = 5400+5500 = 10900
        #   LHR rt = 5700+5600 = 11300
        # Cap at 11000 should keep AMS + CDG and drop LHR.
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
            'same_city': True,
            'max_price': 11000,
        })
        assert result['status'] == 'ok'
        totals = [c['total_price'] for c in result['combinations']]
        assert all(t <= 11000 for t in totals)
        assert 10500 in totals  # AMS round-trip survived
        assert not any(t > 11000 for t in totals)

    def test_max_price_below_everything_yields_no_results(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
            'max_price': 5000,
        })
        assert result['status'] == 'no_results'

    def test_summary_markdown_present(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'western_europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
        })
        md = result['summary_markdown']
        assert 'Option 1' in md
        assert 'Outbound' in md and 'Return' in md


class TestOpenJawEdgeCases:
    def test_missing_required_slots_returns_degrade(self, canned_aggregator):
        result = _run({
            'origin': '',
            'destination_region': 'europe',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
        })
        assert result['status'] == 'error'
        assert result['error_type'] == 'MissingParameterError'

    def test_unknown_region_returns_degrade(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'atlantis',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
        })
        assert result['status'] == 'error'

    def test_comma_separated_iata_list_is_accepted(self, canned_aggregator):
        result = _run({
            'origin': 'HKG',
            'destination_region': 'CDG, FCO, LHR',
            'outbound_date': '2026-04-23',
            'return_date': '2026-05-03',
            'avoid_transit': ['middle_east'],
        })
        assert result['status'] == 'ok'
        assert set(result['candidates']) == {'CDG', 'FCO', 'LHR'}

    def test_all_routes_empty_returns_no_results(self, canned_aggregator):
        # Override the aggregator to one that always raises NoResults
        from agents.errors import NoResultsError as _NRE

        class _Empty(BaseFlightSource):
            name = 'empty'
            def is_configured(self): return True
            def search(self, **kwargs):
                raise _NRE(kwargs.get('origin', ''), kwargs.get('destination', ''),
                           kwargs.get('outbound_date', ''))
            def details(self, fid): return {}

        from agents.data_sources.aggregator import AggregatedFlightSource as _Agg
        import agents.tools.open_jaw as ojs_mod
        import pytest as _pt
        with _pt.MonkeyPatch.context() as mp:
            mp.setattr(ojs_mod, 'get_default_aggregator',
                       lambda: _Agg([_Empty()]))
            result = _run({
                'origin': 'HKG',
                'destination_region': 'europe',
                'outbound_date': '2026-04-23',
                'return_date': '2026-05-03',
            })
        assert result['status'] == 'no_results'
        assert 'Try relaxing' in result['message']
