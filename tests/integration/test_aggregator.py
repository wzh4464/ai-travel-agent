"""Integration tests for the multi-source aggregator.

Covers the fan-out, cross-provider dedupe, partial-failure tolerance, and
the ``FLIGHT_SOURCES`` environment-variable filter.
"""

from __future__ import annotations

import pytest

from agents.data_sources.aggregator import (
    AggregatedFlightSource,
    _dedupe,
    build_default_aggregator,
)
from agents.data_sources.base import BaseFlightSource
from agents.errors import NoResultsError, UpstreamAPIError


def _mk_flight(provider: str, price: float, *, flight_id: str = 'x',
               dep_time: str = '2026-05-01T08:00:00',
               arr_time: str = '2026-05-01T20:00:00',
               airline: str = 'AA') -> dict:
    return {
        'flight_id': flight_id,
        'price': price,
        'currency': 'USD',
        'total_duration_minutes': 720,
        'stops': 0,
        'legs': [{
            'departure_airport': 'JFK',
            'departure_time': dep_time,
            'arrival_airport': 'LHR',
            'arrival_time': arr_time,
            'airline': airline,
        }],
        'provider': provider,
    }


class _StubSource(BaseFlightSource):
    name = 'stub'

    def __init__(self, name: str, *, flights=None, exc: Exception | None = None,
                 configured: bool = True):
        # Intentionally skip BaseFlightSource.__init__ to avoid needing a
        # rate limiter in this toy double.
        self.name = name
        self._flights = flights or []
        self._exc = exc
        self._configured = configured
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return self._configured

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return list(self._flights)

    def details(self, flight_id):
        return {'status': 'unsupported'}


class TestDedupe:
    def test_same_route_keeps_cheapest(self):
        a = _mk_flight('serpapi', 500, flight_id='a')
        b = _mk_flight('amadeus', 420, flight_id='b')
        out = _dedupe([a, b])
        assert len(out) == 1
        assert out[0]['provider'] == 'amadeus'
        assert out[0]['price'] == 420

    def test_different_times_are_kept_separate(self):
        a = _mk_flight('serpapi', 500, dep_time='2026-05-01T08:00:00')
        b = _mk_flight('amadeus', 420, dep_time='2026-05-01T19:00:00')
        out = _dedupe([a, b])
        assert len(out) == 2

    def test_different_airlines_are_kept_separate(self):
        a = _mk_flight('serpapi', 500, airline='AA')
        b = _mk_flight('amadeus', 420, airline='BA')
        out = _dedupe([a, b])
        assert len(out) == 2

    def test_output_sorted_by_price(self):
        flights = [
            _mk_flight('a', 700, airline='X'),
            _mk_flight('b', 500, airline='Y'),
            _mk_flight('c', 600, airline='Z'),
        ]
        out = _dedupe(flights)
        assert [f['price'] for f in out] == [500, 600, 700]

    def test_skips_flights_without_legs(self):
        assert _dedupe([{'legs': [], 'price': 1}]) == []


class TestAggregatorSearch:
    def test_combines_results_from_multiple_sources(self):
        s1 = _StubSource('s1', flights=[_mk_flight('s1', 500, airline='AA')])
        s2 = _StubSource('s2', flights=[_mk_flight('s2', 400, airline='BA')])
        agg = AggregatedFlightSource([s1, s2])
        results = agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
        assert len(results) == 2
        assert {f['provider'] for f in results} == {'s1', 's2'}

    def test_tolerates_partial_failure(self):
        s1 = _StubSource('s1', flights=[_mk_flight('s1', 500)])
        s2 = _StubSource('s2', exc=UpstreamAPIError('s2', status=500, detail='boom'))
        agg = AggregatedFlightSource([s1, s2])
        results = agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
        assert len(results) == 1
        assert results[0]['provider'] == 's1'

    def test_all_failing_raises_upstream_error(self):
        s1 = _StubSource('s1', exc=UpstreamAPIError('s1', status=500))
        s2 = _StubSource('s2', exc=UpstreamAPIError('s2', status=500))
        agg = AggregatedFlightSource([s1, s2])
        with pytest.raises(UpstreamAPIError):
            agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_all_no_results_raises_no_results(self):
        s1 = _StubSource('s1', exc=NoResultsError('JFK', 'LHR', '2026-05-01'))
        s2 = _StubSource('s2', exc=NoResultsError('JFK', 'LHR', '2026-05-01'))
        agg = AggregatedFlightSource([s1, s2])
        with pytest.raises(NoResultsError):
            agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_unconfigured_sources_are_skipped(self):
        s1 = _StubSource('s1', flights=[_mk_flight('s1', 500)], configured=False)
        s2 = _StubSource('s2', flights=[_mk_flight('s2', 400)])
        agg = AggregatedFlightSource([s1, s2])
        results = agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
        assert all(f['provider'] == 's2' for f in results)
        assert s1.calls == []

    def test_parallel_can_be_disabled(self):
        s1 = _StubSource('s1', flights=[_mk_flight('s1', 500, airline='AA')])
        s2 = _StubSource('s2', flights=[_mk_flight('s2', 400, airline='BA')])
        agg = AggregatedFlightSource([s1, s2])
        results = agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01', parallel=False)
        assert len(results) == 2
        assert {f['provider'] for f in results} == {'s1', 's2'}

    def test_no_configured_sources_raises(self):
        s1 = _StubSource('s1', configured=False)
        agg = AggregatedFlightSource([s1])
        with pytest.raises(UpstreamAPIError):
            agg.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')


class TestAggregatorFactory:
    def test_flight_sources_env_filters(self, monkeypatch):
        monkeypatch.setenv('FLIGHT_SOURCES', 'kiwi')
        monkeypatch.setenv('TEQUILA_API_KEY', 'test')
        agg = build_default_aggregator()
        names = [s.name for s in agg.sources]
        assert names == ['kiwi']

    def test_amadeus_credentials_build_source(self, monkeypatch):
        monkeypatch.setenv('AMADEUS_CLIENT_ID', 'id')
        monkeypatch.setenv('AMADEUS_CLIENT_SECRET', 'sec')
        agg = build_default_aggregator()
        assert any(s.name == 'amadeus' for s in agg.active_sources())

    def test_no_env_gives_empty_active(self):
        agg = build_default_aggregator()
        assert agg.active_sources() == []
