"""Integration tests for the Kiwi Tequila adapter."""

from __future__ import annotations

import urllib.parse

import pytest

from agents.data_sources.kiwi_source import KiwiFlightSource, _iso_to_kiwi_date
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError
from tests.integration.http_mock import make_http_error, mock_urlopen

_SEARCH_URL = 'https://api.tequila.kiwi.com/v2/search'


def _parse_qs(request):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(request.full_url).query))


@pytest.fixture
def source():
    return KiwiFlightSource(api_key='test-key')


class TestKiwiConfiguration:
    def test_is_configured_requires_api_key(self):
        assert KiwiFlightSource(api_key='x').is_configured()
        assert not KiwiFlightSource(api_key=None).is_configured()

    def test_search_raises_when_unconfigured(self):
        src = KiwiFlightSource(api_key=None)
        with pytest.raises(UpstreamAPIError):
            src.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_iso_to_kiwi_date_conversion(self):
        assert _iso_to_kiwi_date('2026-05-01') == '01/05/2026'


class TestKiwiSearchHappyPath:
    def test_returns_normalised_flights(self, source, kiwi_search_fixture):
        with mock_urlopen() as mock:
            mock.add('GET', _SEARCH_URL, kiwi_search_fixture)
            results = source.search(
                origin='JFK', destination='LHR', outbound_date='2026-05-01'
            )
        assert len(results) == 2
        assert results[0]['provider'] == 'kiwi'
        assert results[0]['flight_id'] == 'kw-offer-1'
        assert results[0]['price'] == 399.0
        assert results[0]['booking_url'].startswith('https://www.kiwi.com/book/')

    def test_query_string_contains_expected_params(self, source, kiwi_search_fixture):
        with mock_urlopen() as mock:
            mock.add('GET', _SEARCH_URL, kiwi_search_fixture)
            source.search(
                origin='JFK', destination='LHR',
                outbound_date='2026-05-01', return_date='2026-05-08',
                adults=2, cabin_class='business', max_stops=1,
            )

        request = mock.requests[-1]
        params = _parse_qs(request)
        assert params['fly_from'] == 'JFK'
        assert params['fly_to'] == 'LHR'
        assert params['date_from'] == '01/05/2026'
        assert params['date_to'] == '01/05/2026'
        assert params['return_from'] == '08/05/2026'
        assert params['selected_cabins'] == 'C'  # business → C
        assert params['max_stopovers'] == '1'
        assert params['adults'] == '2'

    def test_api_key_in_header(self, source, kiwi_search_fixture):
        with mock_urlopen() as mock:
            mock.add('GET', _SEARCH_URL, kiwi_search_fixture)
            source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
        request = mock.requests[-1]
        # Header casing depends on urllib internals; check both.
        header = request.headers.get('apikey') or request.headers.get('Apikey')
        assert header == 'test-key'


class TestKiwiErrors:
    def test_429_raises_rate_limited(self, source, monkeypatch):
        monkeypatch.setattr('agents.data_sources.base.time.sleep', lambda *_: None)
        with mock_urlopen() as mock:
            for _ in range(4):
                mock.add('GET', _SEARCH_URL, make_http_error(_SEARCH_URL, 429, b'{}'))
            with pytest.raises(RateLimitedError):
                source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_empty_data_raises_no_results(self, source):
        with mock_urlopen() as mock:
            mock.add('GET', _SEARCH_URL, {'data': []})
            with pytest.raises(NoResultsError):
                source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
