"""Integration tests for the Amadeus adapter.

The Amadeus source uses stdlib ``urllib`` — every test in this file patches
``urllib.request.urlopen`` so no network traffic leaves the box. Tests
cover the OAuth2 flow, query-string construction, happy path, the 429
``RateLimitedError`` conversion, and the "no results" path.
"""

from __future__ import annotations

import json
import urllib.parse

import pytest

from agents.data_sources.amadeus_source import AmadeusFlightSource
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError
from tests.integration.http_mock import make_http_error, mock_urlopen

_TOKEN_URL = 'https://test.api.amadeus.com/v1/security/oauth2/token'
_OFFERS_URL = 'https://test.api.amadeus.com/v2/shopping/flight-offers'


def _parse_qs(request):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(request.full_url).query))


@pytest.fixture
def source():
    return AmadeusFlightSource(client_id='test-id', client_secret='test-secret')


class TestAmadeusAuth:
    def test_is_configured_requires_both_credentials(self):
        assert AmadeusFlightSource(client_id='a', client_secret='b').is_configured()
        assert not AmadeusFlightSource(client_id=None, client_secret='b').is_configured()
        assert not AmadeusFlightSource(client_id='a', client_secret=None).is_configured()

    def test_search_raises_when_unconfigured(self):
        src = AmadeusFlightSource(client_id=None, client_secret=None)
        with pytest.raises(UpstreamAPIError):
            src.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_token_is_fetched_and_cached(self, source, amadeus_token_fixture, amadeus_offers_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            mock.add('GET', _OFFERS_URL, amadeus_offers_fixture)
            mock.add('GET', _OFFERS_URL, amadeus_offers_fixture)
            source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
            source.search(origin='JFK', destination='LHR', outbound_date='2026-05-02')
        token_requests = [r for r in mock.requests if r.full_url == _TOKEN_URL]
        assert len(token_requests) == 1  # cached after the first call


class TestAmadeusSearchHappyPath:
    def test_returns_normalised_flights(self, source, amadeus_token_fixture, amadeus_offers_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            mock.add('GET', _OFFERS_URL, amadeus_offers_fixture)
            results = source.search(
                origin='JFK', destination='LHR',
                outbound_date='2026-05-01', adults=2, cabin_class='business',
            )

        assert len(results) == 2
        first = results[0]
        assert first['provider'] == 'amadeus'
        assert first['flight_id'] == '1'
        assert first['price'] == 420.5
        assert first['legs'][0]['airline'] == 'AMERICAN AIRLINES'
        assert first['currency'] == 'USD'

    def test_query_string_contains_expected_params(self, source, amadeus_token_fixture, amadeus_offers_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            mock.add('GET', _OFFERS_URL, amadeus_offers_fixture)
            source.search(
                origin='JFK', destination='LHR',
                outbound_date='2026-05-01', adults=2, children=1,
                cabin_class='business', max_stops=0, return_date='2026-05-08',
            )

        offers_request = next(r for r in mock.requests if 'flight-offers' in r.full_url)
        params = _parse_qs(offers_request)
        assert params['originLocationCode'] == 'JFK'
        assert params['destinationLocationCode'] == 'LHR'
        assert params['departureDate'] == '2026-05-01'
        assert params['returnDate'] == '2026-05-08'
        assert params['adults'] == '2'
        assert params['children'] == '1'
        assert params['travelClass'] == 'BUSINESS'
        assert params['nonStop'] == 'true'
        assert params['currencyCode'] == 'USD'

    def test_authorization_header_is_bearer(self, source, amadeus_token_fixture, amadeus_offers_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            mock.add('GET', _OFFERS_URL, amadeus_offers_fixture)
            source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
        offers_request = next(r for r in mock.requests if 'flight-offers' in r.full_url)
        auth = offers_request.headers.get('Authorization') or offers_request.headers.get('authorization')
        assert auth == f'Bearer {amadeus_token_fixture["access_token"]}'


class TestAmadeusErrors:
    def test_429_becomes_rate_limited_error(self, source, amadeus_token_fixture, monkeypatch):
        # Make retries instant so the test finishes quickly.
        monkeypatch.setattr('agents.data_sources.base.time.sleep', lambda *_: None)

        class _Headers:
            def __init__(self, mapping):
                self._m = mapping
            def get(self, key, default=None):
                return self._m.get(key, default)

        err = make_http_error(_OFFERS_URL, 429, b'{}')
        err.headers = _Headers({'Retry-After': '7'})

        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            mock.add('GET', _OFFERS_URL, err)
            mock.add('GET', _OFFERS_URL, err)
            mock.add('GET', _OFFERS_URL, err)
            mock.add('GET', _OFFERS_URL, err)
            with pytest.raises(RateLimitedError):
                source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_500_becomes_upstream_api_error(self, source, amadeus_token_fixture, monkeypatch):
        # Make retries fast so we do not sleep.
        monkeypatch.setattr('agents.data_sources.base.time.sleep', lambda *_: None)
        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            for _ in range(4):
                mock.add('GET', _OFFERS_URL, make_http_error(_OFFERS_URL, 500, b'{}'))
            with pytest.raises(UpstreamAPIError):
                source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')

    def test_empty_data_becomes_no_results(self, source, amadeus_token_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _TOKEN_URL, amadeus_token_fixture)
            mock.add('GET', _OFFERS_URL, {'data': [], 'dictionaries': {}})
            with pytest.raises(NoResultsError):
                source.search(origin='JFK', destination='LHR', outbound_date='2026-05-01')
