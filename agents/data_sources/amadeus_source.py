"""Amadeus Self-Service ``flight-offers`` adapter.

Uses the stdlib :mod:`urllib` client so the adapter has no runtime
dependencies beyond Python itself. Authentication is the OAuth2
client-credentials flow documented at
https://developers.amadeus.com/self-service/apis-docs/guides/authorization-262.
Tokens are cached in-process and refreshed when close to expiry.

Configure via environment variables:

    AMADEUS_CLIENT_ID        — required
    AMADEUS_CLIENT_SECRET    — required
    AMADEUS_BASE_URL         — optional, defaults to the test sandbox
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agents.data_sources.base import BaseFlightSource, RateLimiter, with_retries
from agents.data_sources.normalizer import normalize_amadeus
from agents.errors import NoResultsError, RateLimitedError, UpstreamAPIError

_AMADEUS_CABIN = {
    'economy': 'ECONOMY',
    'premium_economy': 'PREMIUM_ECONOMY',
    'business': 'BUSINESS',
    'first': 'FIRST',
}

_DEFAULT_BASE_URL = 'https://test.api.amadeus.com'


class AmadeusFlightSource(BaseFlightSource):
    name = 'amadeus'

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 20.0,
    ) -> None:
        super().__init__(rate_limiter=rate_limiter or RateLimiter(rate_per_second=5.0, burst=10))
        self._client_id = client_id or os.environ.get('AMADEUS_CLIENT_ID')
        self._client_secret = client_secret or os.environ.get('AMADEUS_CLIENT_SECRET')
        self._base_url = (
            base_url
            or os.environ.get('AMADEUS_BASE_URL')
            or _DEFAULT_BASE_URL
        ).rstrip('/')
        self._timeout = timeout
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # configuration & auth
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 30:
            return self._token
        if not self.is_configured():
            raise UpstreamAPIError(self.name, detail='AMADEUS_CLIENT_ID/SECRET not set')

        body = urllib.parse.urlencode(
            {
                'grant_type': 'client_credentials',
                'client_id': self._client_id,
                'client_secret': self._client_secret,
            }
        ).encode('utf-8')
        req = urllib.request.Request(
            f'{self._base_url}/v1/security/oauth2/token',
            data=body,
            method='POST',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise UpstreamAPIError(
                self.name,
                status=exc.code,
                detail='token request failed',
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamAPIError(self.name, detail=f'network error: {exc.reason}') from exc

        token = payload.get('access_token')
        if not token:
            raise UpstreamAPIError(self.name, detail='no access_token in response')
        self._token = token
        self._token_expires_at = time.time() + float(payload.get('expires_in', 0) or 0)
        return token

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

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
        self.rate_limiter.acquire()
        token = self._get_token()

        params: dict[str, Any] = {
            'originLocationCode': origin,
            'destinationLocationCode': destination,
            'departureDate': outbound_date,
            'adults': adults,
            'travelClass': _AMADEUS_CABIN.get(cabin_class.lower(), 'ECONOMY'),
            'currencyCode': 'USD',
            'max': 20,
        }
        if children:
            params['children'] = children
        infants = (infants_in_seat or 0) + (infants_on_lap or 0)
        if infants:
            params['infants'] = infants
        if return_date:
            params['returnDate'] = return_date
        if max_stops == 0:
            params['nonStop'] = 'true'

        data = self._get_json('/v2/shopping/flight-offers', params, token)
        offers = data.get('data', []) or []
        carriers = ((data.get('dictionaries') or {}).get('carriers') or {})
        if not offers:
            raise NoResultsError(origin, destination, outbound_date)
        return [normalize_amadeus(o, carriers, provider=self.name).to_dict() for o in offers]

    @with_retries()
    def details(self, flight_id: str) -> dict:
        """The Self-Service offer payload already contains all details; there
        is no per-id lookup endpoint. Callers should pass the full flight dict
        into :func:`agents.tools.flight_details.get_flight_details`.
        """
        return {
            'status': 'unsupported',
            'message': 'Amadeus offers are fully described by the search payload.',
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _get_json(self, path: str, params: dict[str, Any], token: str) -> dict:
        qs = urllib.parse.urlencode(params)
        url = f'{self._base_url}{path}?{qs}'
        req = urllib.request.Request(
            url,
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json',
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
