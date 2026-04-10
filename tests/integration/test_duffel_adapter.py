"""Integration tests for the Duffel adapter.

Covers request shape (POST body, Authorization, Duffel-Version header),
offer parsing, 429 → RateLimitedError, empty results, and max_stops
client-side filtering.
"""

from __future__ import annotations

import json

import pytest

from agents.data_sources.duffel_source import DuffelFlightSource
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError
from tests.integration.http_mock import make_http_error, mock_urlopen

_OFFER_REQUESTS_URL = 'https://api.duffel.com/air/offer_requests'


@pytest.fixture
def source():
    return DuffelFlightSource(api_key='duffel_test_FAKE-CREDENTIAL-FOR-UNIT-TESTS')


class TestConfiguration:
    def test_is_configured_requires_api_key(self):
        assert DuffelFlightSource(api_key='x').is_configured()
        assert not DuffelFlightSource(api_key=None).is_configured()

    def test_search_raises_when_unconfigured(self):
        src = DuffelFlightSource(api_key=None)
        with pytest.raises(UpstreamAPIError):
            src.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')


class TestHappyPath:
    def test_returns_normalised_flights(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            results = source.search(
                origin='HKG', destination='CDG',
                outbound_date='2026-04-23', adults=1, cabin_class='economy',
            )

        assert len(results) == 2
        assert all(f['provider'] == 'duffel' for f in results)
        first = results[0]
        assert first['flight_id'] == 'off_0000AaaaNonstop'
        assert first['price'] == 5480.0
        assert first['currency'] == 'HKD'
        assert first['stops'] == 0

    def test_request_url_has_return_offers_flag(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')
        req = mock.requests[-1]
        assert 'return_offers=true' in req.full_url

    def test_duffel_version_header_present(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')
        req = mock.requests[-1]
        version = req.headers.get('Duffel-version') or req.headers.get('Duffel-Version')
        assert version == 'v2'

    def test_authorization_bearer_token(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')
        req = mock.requests[-1]
        auth = req.headers.get('Authorization') or req.headers.get('authorization')
        assert auth.startswith('Bearer duffel_test_')

    def test_body_contains_slices_and_passengers(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            source.search(
                origin='HKG', destination='CDG',
                outbound_date='2026-04-23', return_date='2026-05-03',
                adults=2, cabin_class='business',
            )
        req = mock.requests[-1]
        body = json.loads(req.data.decode('utf-8'))

        # Outbound + return = 2 slices
        slices = body['data']['slices']
        assert len(slices) == 2
        assert slices[0] == {
            'origin': 'HKG', 'destination': 'CDG', 'departure_date': '2026-04-23'
        }
        assert slices[1] == {
            'origin': 'CDG', 'destination': 'HKG', 'departure_date': '2026-05-03'
        }

        passengers = body['data']['passengers']
        assert len(passengers) == 2
        assert all(p['type'] == 'adult' for p in passengers)

        assert body['data']['cabin_class'] == 'business'

    def test_one_way_has_single_slice(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')
        req = mock.requests[-1]
        body = json.loads(req.data.decode('utf-8'))
        assert len(body['data']['slices']) == 1


class TestMaxStopsClientSideFilter:
    def test_max_stops_zero_filters_one_stop_offers(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            results = source.search(
                origin='HKG', destination='CDG',
                outbound_date='2026-04-23', max_stops=0,
            )
        assert len(results) == 1
        assert results[0]['stops'] == 0

    def test_max_stops_none_returns_everything(self, source, duffel_offer_request_fixture):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, duffel_offer_request_fixture)
            results = source.search(
                origin='HKG', destination='CDG', outbound_date='2026-04-23',
            )
        assert len(results) == 2


class TestErrors:
    def test_429_raises_rate_limited(self, source, monkeypatch):
        monkeypatch.setattr('agents.data_sources.base.time.sleep', lambda *_: None)
        with mock_urlopen() as mock:
            for _ in range(4):
                mock.add(
                    'POST', _OFFER_REQUESTS_URL,
                    make_http_error(_OFFER_REQUESTS_URL, 429, b'{}'),
                )
            with pytest.raises(RateLimitedError):
                source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')

    def test_422_validation_error_is_upstream(self, source, monkeypatch):
        monkeypatch.setattr('agents.data_sources.base.time.sleep', lambda *_: None)
        with mock_urlopen() as mock:
            for _ in range(4):
                mock.add(
                    'POST', _OFFER_REQUESTS_URL,
                    make_http_error(_OFFER_REQUESTS_URL, 422, b'{"errors": []}'),
                )
            with pytest.raises(UpstreamAPIError):
                source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')

    def test_empty_offers_raises_no_results(self, source):
        with mock_urlopen() as mock:
            mock.add('POST', _OFFER_REQUESTS_URL, {'data': {'offers': []}})
            with pytest.raises(NoResultsError):
                source.search(origin='HKG', destination='CDG', outbound_date='2026-04-23')
