"""Multi-source aggregator.

Fans a single canonical search request out to every configured data source
in parallel, merges the normalised results, and de-duplicates flights that
appear in more than one provider (keeping the cheapest copy).

Partial failures are tolerated: if at least one source returns results the
aggregator returns them. Only when every source fails do we surface an
error to the caller.
"""

from __future__ import annotations

import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from agents.data_sources.base import BaseFlightSource
from agents.errors import NoResultsError, TravelAgentError, UpstreamAPIError

logger = logging.getLogger(__name__)


def _canon_code(value) -> str:
    """Uppercase + strip a flight number or airport code so providers that
    pad / lowercase / annotate identifiers still collide in dedupe."""
    return (str(value or '')).strip().upper()


def _canon_time(value) -> str:
    """Truncate ISO-8601 timestamps to whole minutes so providers reporting
    different second-level precision still collide on the same departure.
    Times that don't parse as ISO fall back to their original string."""
    s = str(value or '').strip()
    if not s:
        return ''
    try:
        # Strip trailing Z, accept missing seconds, normalise to UTC if a
        # timezone offset is present — we only care about the wall-clock
        # minute for dedupe.
        normalised = s.replace('Z', '+00:00')
        dt = datetime.datetime.fromisoformat(normalised)
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt.replace(second=0, microsecond=0).isoformat(timespec='minutes')
    except ValueError:
        return s


def _dedupe_price(flight: dict) -> float:
    """Return a sortable price; missing/zero/garbage become +inf so they
    can never replace a real priced duplicate."""
    raw = flight.get('price')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float('inf')
    if value <= 0:
        return float('inf')
    return value


def _dedupe(flights: list[dict]) -> list[dict]:
    """Collapse duplicates across providers, keeping the cheapest entry.

    The key normalises code casing and timestamp precision so the same
    leg surfaced by two providers actually collides. Missing / zero prices
    are treated as +inf so an incomplete provider entry can never replace
    a real priced duplicate.
    """
    by_key: dict[tuple, dict] = {}
    for flight in flights:
        legs = flight.get('legs') or []
        if not legs:
            continue
        # Airline strings are not canonical across providers (Amadeus uses
        # display names, Kiwi uses 2-letter codes), so they're deliberately
        # excluded — flight_number already encodes the carrier code prefix.
        leg_sequence = tuple(
            (
                _canon_code(leg.get('flight_number')),
                _canon_code(leg.get('departure_airport')),
                _canon_time(leg.get('departure_time')),
                _canon_code(leg.get('arrival_airport')),
                _canon_time(leg.get('arrival_time')),
            )
            for leg in legs
        )
        # ``currency`` is part of the key so the same physical flight quoted
        # in two currencies (e.g. SerpAPI/USD and Duffel/HKD for the route
        # HKG→CDG) is NOT collapsed: the prices are not comparable as raw
        # numbers, and dropping one would mislead downstream ranking.
        key = (
            _canon_code(legs[0].get('departure_airport')),
            _canon_time(legs[0].get('departure_time')),
            _canon_code(legs[-1].get('arrival_airport')),
            _canon_time(legs[-1].get('arrival_time')),
            len(legs),
            (flight.get('currency') or '').upper(),
            leg_sequence,
        )
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = flight
            continue
        if _dedupe_price(flight) < _dedupe_price(existing):
            by_key[key] = flight
    return sorted(by_key.values(), key=_dedupe_price)


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
        succeeded: set[str] = set()
        with ThreadPoolExecutor(max_workers=min(4, len(active))) as pool:
            future_to_source = {pool.submit(s.search, **kwargs): s for s in active}
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    payload = future.result() or []
                    succeeded.add(source.name)
                    results.extend(payload)
                except NoResultsError as exc:
                    # A clean "no flights" answer still counts as a successful
                    # roundtrip for the purpose of "did anyone respond?".
                    succeeded.add(source.name)
                    errors.append((source.name, exc))
                except TravelAgentError as exc:
                    errors.append((source.name, exc))
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append((source.name, UpstreamAPIError(source.name, detail=str(exc))))

        if results:
            return _dedupe(results)

        # No flights came back from anyone. Decide whether that's a clean
        # "no results" answer or an upstream outage.
        if succeeded:
            # At least one source responded successfully (with [] or
            # NoResultsError). The route legitimately has no flights —
            # even if other sources errored, we have a real answer.
            raise NoResultsError(origin, destination, outbound_date)
        summary = '; '.join(f'{name}: {err}' for name, err in errors)
        raise UpstreamAPIError(self.name, detail=f'all sources failed ({summary})')

    def details(self, flight_id: str) -> dict:
        for source in self.active_sources():
            try:
                payload = source.details(flight_id)
            except TravelAgentError as exc:
                logger.debug(
                    'aggregator.details: %s failed for %s: %s',
                    source.name, flight_id, exc,
                )
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
    except (ImportError, ModuleNotFoundError):
        # Optional dep not installed (e.g. ``serpapi`` package missing).
        # Aggregator construction must still succeed.
        return None
    except Exception as exc:  # pylint: disable=broad-except
        # A configured-but-broken source should not crash the whole agent,
        # but it must be diagnosable — log it instead of swallowing.
        logger.warning('flight source %r failed to initialise: %s', name, exc)
        return None
    return instance


def build_default_aggregator() -> AggregatedFlightSource:
    """Assemble an aggregator from every source whose optional dependency loads.

    Sources are *constructed* eagerly (the SDK / optional dep has to import)
    but the credential check is deferred to ``is_configured()``, which the
    aggregator consults via ``active_sources()`` on every call. That keeps
    runtime configuration (env vars rotated mid-process, late ``.env``
    loading) actually live.
    """
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

    for name, factory in (('serpapi', _serpapi), ('amadeus', _amadeus), ('kiwi', _kiwi)):
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
