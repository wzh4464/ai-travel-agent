"""SerpAPI (Google Flights) data source implementation."""

from __future__ import annotations

import os
from typing import Any

import serpapi

from agents.data_sources.base import BaseFlightSource, RateLimiter, with_retries
from agents.data_sources.normalizer import Flight, normalize_serpapi
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError

_CABIN_MAP = {
    'economy': 1,
    'premium_economy': 2,
    'business': 3,
    'first': 4,
}


class SerpAPIFlightSource(BaseFlightSource):
    name = 'serpapi-google-flights'

    def __init__(
        self,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        super().__init__(rate_limiter=rate_limiter or RateLimiter(rate_per_second=1.5, burst=3))
        self._api_key = api_key or os.environ.get('SERPAPI_API_KEY')

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _raw_search(self, params: dict[str, Any]) -> dict:
        if not self._api_key:
            raise UpstreamAPIError(self.name, detail='SERPAPI_API_KEY is not set')
        self.rate_limiter.acquire()
        params = {**params, 'api_key': self._api_key}
        try:
            return serpapi.search(params).data
        except Exception as exc:  # serpapi raises bare Exception subclasses
            msg = str(exc).lower()
            if '429' in msg or 'rate' in msg:
                raise RateLimitedError(self.name) from exc
            raise UpstreamAPIError(self.name, detail=str(exc)) from exc

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
        params: dict[str, Any] = {
            'engine': 'google_flights',
            'hl': 'en',
            'gl': 'us',
            'currency': 'USD',
            'departure_id': origin,
            'arrival_id': destination,
            'outbound_date': outbound_date,
            'adults': adults,
            'children': children,
            'infants_in_seat': infants_in_seat,
            'infants_on_lap': infants_on_lap,
            'travel_class': _CABIN_MAP.get(cabin_class.lower(), 1),
        }
        if return_date:
            params['return_date'] = return_date
        else:
            # Google Flights via SerpAPI defaults to round-trip (type=1).
            # Without this flag a one-way request silently returns no results
            # because the API expects a return_date.
            params['type'] = 2
        if max_stops is not None:
            # SerpAPI: 0=any, 1=non-stop, 2=<=1 stop, 3=<=2 stops
            params['stops'] = {0: 1, 1: 2, 2: 3}.get(max_stops, 0)

        data = self._raw_search(params)
        raw_flights = (data.get('best_flights') or []) + (data.get('other_flights') or [])
        if not raw_flights:
            raise NoResultsError(origin, destination, outbound_date)
        return [normalize_serpapi(f, provider=self.name).to_dict() for f in raw_flights]

    @with_retries()
    def details(self, flight_id: str) -> dict:
        """SerpAPI does not expose a per-flight lookup; the details live on the
        search response itself. Callers are expected to pass the full Flight
        dict back in via :func:`agents.tools.flight_details.get_flight_details`.
        """
        return {
            'status': 'unsupported',
            'message': 'Per-ID lookup is not available for this provider.',
            'hint': 'Re-run search and read the details from the flight payload.',
        }


_default_source: SerpAPIFlightSource | None = None


def get_default_source() -> SerpAPIFlightSource:
    """Return a process-wide SerpAPI source instance (lazy singleton)."""
    global _default_source
    if _default_source is None:
        _default_source = SerpAPIFlightSource()
    return _default_source
