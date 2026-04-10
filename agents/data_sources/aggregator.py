"""Multi-source aggregator.

Fans a single canonical search request out to every configured data source
in parallel, merges the normalised results, and de-duplicates flights that
appear in more than one provider (keeping the cheapest copy).

Partial failures are tolerated: if at least one source returns results the
aggregator returns them. Only when every source fails do we surface an
error to the caller.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from agents.data_sources.base import BaseFlightSource
from agents.errors import NoResultsError, TravelAgentError, UpstreamAPIError


def _dedupe(flights: list[dict]) -> list[dict]:
    """Collapse duplicates across providers, keeping the cheapest entry."""
    by_key: dict[tuple, dict] = {}
    for flight in flights:
        legs = flight.get('legs') or []
        if not legs:
            continue
        key = (
            legs[0].get('departure_airport'),
            legs[0].get('departure_time'),
            legs[-1].get('arrival_airport'),
            legs[-1].get('arrival_time'),
            legs[0].get('airline'),
        )
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = flight
            continue
        if float(flight.get('price') or 0) < float(existing.get('price') or 0):
            by_key[key] = flight
    return sorted(by_key.values(), key=lambda f: float(f.get('price') or 10**9))


class AggregatedFlightSource(BaseFlightSource):
    """Fan-out wrapper that behaves like a single BaseFlightSource."""

    name = 'aggregator'

    def __init__(self, sources: Iterable[BaseFlightSource]):
        # Intentionally skip BaseFlightSource.__init__ — the aggregator never
        # makes its own HTTP calls so it does not need a rate limiter.
        self._sources: list[BaseFlightSource] = list(sources)

    # ------------------------------------------------------------------

    @property
    def sources(self) -> list[BaseFlightSource]:
        return list(self._sources)

    def active_sources(self) -> list[BaseFlightSource]:
        return [s for s in self._sources if s.is_configured()]

    def is_configured(self) -> bool:
        return any(s.is_configured() for s in self._sources)

    # ------------------------------------------------------------------

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
        active = self.active_sources()
        if not active:
            raise UpstreamAPIError(
                self.name,
                detail='No flight data sources are configured. Set SERPAPI_API_KEY, '
                'AMADEUS_CLIENT_ID/SECRET, or TEQUILA_API_KEY.',
            )

        kwargs = dict(
            origin=origin,
            destination=destination,
            outbound_date=outbound_date,
            return_date=return_date,
            adults=adults,
            children=children,
            infants_in_seat=infants_in_seat,
            infants_on_lap=infants_on_lap,
            cabin_class=cabin_class,
            max_stops=max_stops,
        )

        results: list[dict] = []
        errors: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(active))) as pool:
            future_to_source = {pool.submit(s.search, **kwargs): s for s in active}
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    results.extend(future.result() or [])
                except TravelAgentError as exc:
                    errors.append((source.name, exc))
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append((source.name, UpstreamAPIError(source.name, detail=str(exc))))

        if results:
            return _dedupe(results)

        # Every source failed — surface the most informative error.
        if errors and all(isinstance(err, NoResultsError) for _, err in errors):
            raise NoResultsError(origin, destination, outbound_date)
        summary = '; '.join(f'{name}: {err}' for name, err in errors)
        raise UpstreamAPIError(self.name, detail=f'all sources failed ({summary})')

    def details(self, flight_id: str) -> dict:
        for source in self.active_sources():
            try:
                payload = source.details(flight_id)
            except TravelAgentError:
                continue
            if payload and payload.get('status') != 'unsupported':
                return payload
        return {
            'status': 'unsupported',
            'message': 'No configured data source supports per-id detail lookup.',
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _env_source_filter() -> set[str] | None:
    """Optional ``FLIGHT_SOURCES=serpapi,amadeus,kiwi`` override."""
    raw = os.environ.get('FLIGHT_SOURCES', '').strip()
    if not raw:
        return None
    return {part.strip().lower() for part in raw.split(',') if part.strip()}


def _try_source(factory, name: str, wanted: set[str] | None) -> BaseFlightSource | None:
    if wanted is not None and name not in wanted:
        return None
    try:
        instance = factory()
    except Exception:  # pylint: disable=broad-except
        # Missing optional deps (e.g. ``serpapi`` package not installed) must
        # not break aggregator construction.
        return None
    return instance


def build_default_aggregator() -> AggregatedFlightSource:
    """Assemble an aggregator from every source whose credentials are present."""
    wanted = _env_source_filter()
    sources: list[BaseFlightSource] = []

    def _serpapi():
        from agents.data_sources.serpapi_source import SerpAPIFlightSource  # noqa: WPS433
        return SerpAPIFlightSource()

    def _amadeus():
        from agents.data_sources.amadeus_source import AmadeusFlightSource  # noqa: WPS433
        return AmadeusFlightSource()

    def _kiwi():
        from agents.data_sources.kiwi_source import KiwiFlightSource  # noqa: WPS433
        return KiwiFlightSource()

    def _duffel():
        from agents.data_sources.duffel_source import DuffelFlightSource  # noqa: WPS433
        return DuffelFlightSource()

    for name, factory in (
        ('serpapi', _serpapi),
        ('amadeus', _amadeus),
        ('kiwi', _kiwi),
        ('duffel', _duffel),
    ):
        src = _try_source(factory, name, wanted)
        if src is not None:
            sources.append(src)

    return AggregatedFlightSource(sources)


_default_aggregator: AggregatedFlightSource | None = None


def get_default_aggregator() -> AggregatedFlightSource:
    """Process-wide lazy singleton for the aggregator."""
    global _default_aggregator
    if _default_aggregator is None:
        _default_aggregator = build_default_aggregator()
    return _default_aggregator
