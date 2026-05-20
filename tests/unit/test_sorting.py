"""Unit tests for presentation sorting & filtering helpers."""

from __future__ import annotations

from agents.data_sources.normalizer import normalize_serpapi
from agents.presentation.sorting import filter_flights, sort_flights


def _flights(fixture):
    return [normalize_serpapi(raw).to_dict() for raw in fixture['best_flights']]


class TestSortFlights:
    def test_sort_by_price_ascending(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        ranked = sort_flights(flights, key='price')
        prices = [f['price'] for f in ranked]
        assert prices == sorted(prices)
        assert ranked[0]['price'] == 410.0

    def test_sort_by_duration(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        ranked = sort_flights(flights, key='duration')
        durations = [f['total_duration_minutes'] for f in ranked]
        assert durations == sorted(durations)

    def test_sort_by_stops_then_price(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        ranked = sort_flights(flights, key='stops')
        # Non-stop comes first
        assert ranked[0]['stops'] == 0

    def test_sort_fallback_for_unknown_key(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        ranked = sort_flights(flights, key='not-a-real-key')
        assert ranked[0]['price'] == min(f['price'] for f in flights)

    def test_sort_empty_list(self):
        assert sort_flights([], key='price') == []


class TestFilterFlights:
    def test_filter_by_max_price(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        kept = filter_flights(flights, max_price=420)
        assert len(kept) == 1
        assert kept[0]['price'] == 410.0

    def test_filter_by_max_stops(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        kept = filter_flights(flights, max_stops=0)
        assert len(kept) == 1
        assert kept[0]['stops'] == 0

    def test_filter_by_airline_substring(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        kept = filter_flights(flights, airlines=['american'])
        assert len(kept) == 1
        assert 'American' in kept[0]['legs'][0]['airline']

    def test_filter_combines_all(self, serpapi_flights_fixture):
        flights = _flights(serpapi_flights_fixture)
        kept = filter_flights(flights, max_price=1000, max_stops=2, airlines=['delta'])
        assert len(kept) == 1

    def test_filter_on_empty_list(self):
        assert filter_flights([], max_price=100) == []
