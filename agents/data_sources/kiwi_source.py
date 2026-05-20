"""Kiwi.com Tequila ``/v2/search`` adapter.

API documentation: https://tequila.kiwi.com/portal/docs/tequila_api.

Configure via environment variables:

    TEQUILA_API_KEY          — required (also accepts KIWI_API_KEY)
    KIWI_BASE_URL            — optional, defaults to the public Tequila host
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agents.data_sources.base import BaseFlightSource, RateLimiter, with_retries
from agents.data_sources.normalizer import normalize_kiwi
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError

_KIWI_CABIN = {
    'economy': 'M',
    'premium_economy': 'W',
    'business': 'C',
    'first': 'F',
}

_DEFAULT_BASE_URL = 'https://api.tequila.kiwi.com'


def _iso_to_kiwi_date(iso_date: str) -> str:
    """Kiwi expects ``DD/MM/YYYY`` while the rest of the agent uses ISO."""
    try:
        y, m, d = iso_date.split('-')
    except ValueError:
        return iso_date
    return f'{d}/{m}/{y}'


class KiwiFlightSource(BaseFlightSource):
    name = 'kiwi'

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 20.0,
    ) -> None:
        super().__init__(rate_limiter=rate_limiter or RateLimiter(rate_per_second=2.0, burst=4))
        self._api_key = (
            api_key
            or os.environ.get('TEQUILA_API_KEY')
            or os.environ.get('KIWI_API_KEY')
        )
        self._base_url = (
            base_url
            or os.environ.get('KIWI_BASE_URL')
            or _DEFAULT_BASE_URL
        ).rstrip('/')
        self._timeout = timeout

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
            raise UpstreamAPIError(
                self.name,
                detail='Kiwi API key not set (TEQUILA_API_KEY or KIWI_API_KEY)',
            )
        self.rate_limiter.acquire()

        params: dict[str, Any] = {
            'fly_from': origin,
            'fly_to': destination,
            'date_from': _iso_to_kiwi_date(outbound_date),
            'date_to': _iso_to_kiwi_date(outbound_date),
            'adults': adults,
            'children': children,
            'infants': (infants_in_seat or 0) + (infants_on_lap or 0),
            'selected_cabins': _KIWI_CABIN.get(cabin_class.lower(), 'M'),
            'curr': 'USD',
            'limit': 20,
        }
        if return_date:
            params['return_from'] = _iso_to_kiwi_date(return_date)
            params['return_to'] = _iso_to_kiwi_date(return_date)
        if max_stops is not None:
            params['max_stopovers'] = int(max_stops)

        data = self._get_json('/v2/search', params)
        offers = data.get('data', []) or []
        if not offers:
            raise NoResultsError(origin, destination, outbound_date)
        return [normalize_kiwi(o, provider=self.name).to_dict() for o in offers]

    @with_retries()
    def details(self, flight_id: str) -> dict:
        return {
            'status': 'unsupported',
            'message': 'Use the deep_link field from the Kiwi search payload.',
        }

    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, Any]) -> dict:
        qs = urllib.parse.urlencode(params)
        url = f'{self._base_url}{path}?{qs}'
        req = urllib.request.Request(
            url,
            headers={'apikey': self._api_key or '', 'Accept': 'application/json'},
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
