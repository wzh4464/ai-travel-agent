"""Duffel Air API adapter.

Duffel follows the NDC ``Offer Request → Offers → Orders`` model.
A flight search is implemented as a POST to ``/air/offer_requests`` with
an array of "slices" (one per direction). Passing ``?return_offers=true``
asks Duffel to include the resulting offers inline in the response so we
do not have to make a second GET call.

Configure via environment variables:

    DUFFEL_API_KEY     — required (test keys start with ``duffel_test_``)
    DUFFEL_BASE_URL    — optional, defaults to the production host
                         (the same host is used for both test and live;
                         the key prefix decides the mode)

No extra dependencies: this adapter uses only stdlib ``urllib`` and
``json`` so it can run anywhere the rest of the agent runs.

Reference: https://duffel.com/docs/api/offer-requests
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agents.data_sources.base import BaseFlightSource, RateLimiter, with_retries
from agents.data_sources.normalizer import normalize_duffel
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError

_DEFAULT_BASE_URL = 'https://api.duffel.com'
_API_VERSION = 'v2'

_DUFFEL_CABIN = {
    'economy': 'economy',
    'premium_economy': 'premium_economy',
    'business': 'business',
    'first': 'first',
}


class DuffelFlightSource(BaseFlightSource):
    name = 'duffel'

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 30.0,
    ) -> None:
        # Duffel test keys are generous with bursts but the shared sandbox
        # hosts rate-limit to a few requests per second. 5 rps / burst 10
        # keeps fan-out searches well within budget.
        super().__init__(rate_limiter=rate_limiter or RateLimiter(rate_per_second=5.0, burst=10))
        self._api_key = api_key or os.environ.get('DUFFEL_API_KEY')
        self._base_url = (
            base_url
            or os.environ.get('DUFFEL_BASE_URL')
            or _DEFAULT_BASE_URL
        ).rstrip('/')
        self._timeout = timeout

    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        return bool(self._api_key)

    @with_retries()
    def search(
        self,
        *,
        origin: str,
        destination: str,
        outbound_date: str,
        return_date: str | None = None,
        adults: int = 1,
        children: int = 0,
        infants_in_seat: int = 0,
        infants_on_lap: int = 0,
        cabin_class: str = 'economy',
        max_stops: int | None = None,
    ) -> list[dict]:
        if not self.is_configured():
            raise UpstreamAPIError(self.name, detail='DUFFEL_API_KEY not set')
        self.rate_limiter.acquire()

        slices: list[dict[str, Any]] = [
            {
                'origin': origin,
                'destination': destination,
                'departure_date': outbound_date,
            }
        ]
        if return_date:
            slices.append(
                {
                    'origin': destination,
                    'destination': origin,
                    'departure_date': return_date,
                }
            )

        passengers: list[dict[str, str]] = [{'type': 'adult'} for _ in range(max(1, adults))]
        passengers.extend({'type': 'child'} for _ in range(max(0, children)))
        passengers.extend(
            {'type': 'infant_without_seat'}
            for _ in range(max(0, (infants_in_seat or 0) + (infants_on_lap or 0)))
        )

        body = {
            'data': {
                'slices': slices,
                'passengers': passengers,
                'cabin_class': _DUFFEL_CABIN.get(cabin_class.lower(), 'economy'),
            }
        }

        data = self._post_json('/air/offer_requests?return_offers=true', body)
        offers = ((data.get('data') or {}).get('offers') or [])
        if not offers:
            raise NoResultsError(origin, destination, outbound_date)

        flights = [normalize_duffel(offer, provider=self.name).to_dict() for offer in offers]

        # Duffel doesn't have a native "max stops" filter, so we apply it
        # client-side after normalisation.
        if max_stops is not None:
            flights = [f for f in flights if (f.get('stops') or 0) <= max_stops]
            if not flights:
                raise NoResultsError(origin, destination, outbound_date)
        return flights

    @with_retries()
    def details(self, flight_id: str) -> dict:
        return {
            'status': 'unsupported',
            'message': 'Duffel offers are fully described by the search payload.',
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _post_json(self, path: str, body: dict[str, Any]) -> dict:
        url = f'{self._base_url}{path}'
        encoded = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=encoded,
            method='POST',
            headers={
                'Authorization': f'Bearer {self._api_key}',
                'Duffel-Version': _API_VERSION,
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = ''
            try:
                detail = exc.read().decode('utf-8', 'replace')[:200]
            except Exception:  # pylint: disable=broad-except
                pass
            if exc.code == 429:
                retry_after = None
                try:
                    retry_after = float(exc.headers.get('Retry-After') or 0) or None
                except (TypeError, ValueError):
                    retry_after = None
                raise RateLimitedError(self.name, retry_after=retry_after) from exc
            raise UpstreamAPIError(self.name, status=exc.code, detail=detail) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(self.name, detail=f'network error: {exc.reason}') from exc
