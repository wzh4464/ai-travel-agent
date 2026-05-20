"""Open-jaw / flexible-destination search orchestrator.

Answers "I want to go to Europe on these dates, where in Europe doesn't
matter, show me the cheapest itinerary that doesn't transit through
Middle East hubs" in one tool call.

Orchestration
-------------

1. Expand the user's region name into a list of candidate airports.
2. Parallel fan-out: one one-way search per (origin, candidate_city)
   for the outbound direction, and one per (candidate_city, origin) for
   the return direction.
3. Filter out flights that transit through banned airports.
4. Build a cartesian product of outbound × return and rank by total
   combined price.
5. Return the top N combinations as canonical data plus a pre-rendered
   Markdown summary.

All logic here is pure once the search calls come back — the ranking
and formatting is tested by ``tests/unit/test_open_jaw_ranking.py``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from agents._pydantic_compat import BaseModel, Field
from agents.data_sources import get_default_aggregator
from agents.data_sources.aggregator import AggregatedFlightSource
from agents.errors import (
    InvalidParameterError,
    MissingParameterError,
    NoResultsError,
    TravelAgentError,
    degrade,
)
from agents.presentation.itinerary import (
    format_open_jaw_combinations,
    rank_open_jaw_combinations,
)
from agents.regions import expand_region, expand_transit_blacklist

try:  # langchain is an optional dep for unit tests
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    def tool(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) is False else decorator

logger = logging.getLogger(__name__)


class OpenJawInput(BaseModel):
    origin: str = Field(description='Origin airport code (IATA), e.g. "HKG".')
    destination_region: str = Field(
        description=(
            'A region name ("europe", "western_europe", "northern_europe", '
            '"southern_europe", "central_europe", "欧洲", "北欧", ...) or a '
            'comma-separated list of airport codes like "LHR,CDG,FCO".'
        ),
    )
    outbound_date: str = Field(description='Outbound date in YYYY-MM-DD.')
    return_date: str = Field(description='Return date in YYYY-MM-DD.')
    adults: Optional[int] = Field(1, description='Number of adult passengers.')
    cabin_class: Optional[str] = Field(
        'economy',
        description='economy | premium_economy | business | first.',
    )
    avoid_transit: Optional[List[str]] = Field(
        default=None,
        description=(
            'Blacklist names or raw airport codes to avoid as *transit* '
            'points. Pass "middle_east" for DXB/DOH/AUH/RUH/KWI/BAH/MCT, '
            '"middle_east_strict" to also include IST/SAW, or raw IATA '
            'codes like ["DXB","DOH"] for a custom list.'
        ),
    )
    same_city: Optional[bool] = Field(
        default=False,
        description='Require the return to depart from the same city as the outbound arrival.',
    )
    top_n: Optional[int] = Field(
        default=10,
        description='Maximum number of itinerary combinations to return.',
    )
    max_price: Optional[float] = Field(
        default=None,
        description='Optional cap on the total combined outbound+return price.',
    )


class OpenJawSchema(BaseModel):
    params: OpenJawInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_destinations(destination_region: str) -> list[str]:
    expanded = expand_region(destination_region)
    if expanded:
        return list(expanded)
    # Fall back to a comma-separated list of IATA codes. Only accept
    # tokens that look like real 3-letter codes so a bogus string like
    # "atlantis" does not silently become a single-element list.
    codes: list[str] = []
    for raw in (destination_region or '').split(','):
        token = raw.strip().upper()
        if len(token) == 3 and token.isalpha():
            codes.append(token)
    return codes


def _resolve_banned(avoid: list[str] | None) -> set[str]:
    if not avoid:
        return set()
    banned: set[str] = set()
    for raw in avoid:
        if not raw:
            continue
        banned |= expand_transit_blacklist(raw)
    return banned


def _fan_out(
    source,
    pairs: list[tuple[str, str]],
    *,
    date: str,
    adults: int,
    cabin: str,
    max_workers: int = 4,
) -> tuple[dict[tuple[str, str], list[dict]], list[Exception]]:
    """Run one-way searches for every (origin, destination) pair in parallel.

    Individual pair failures degrade to empty results so that one flaky leg
    does not sink the whole search. Errors are returned so callers can detect
    the all-failed case (and surface it instead of a misleading "no results").

    ``max_workers`` is intentionally modest: ``source.search()`` is typically
    the aggregator, which spawns its own ThreadPoolExecutor across providers.
    Total concurrency is therefore ``pairs`` × ``providers``, so keeping the
    outer pool tight (≤4) avoids the multiplicative thread blow-up flagged
    in reviews.
    """
    results: dict[tuple[str, str], list[dict]] = {pair: [] for pair in pairs}
    errors: list[Exception] = []
    if not pairs:
        return results, errors

    with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs))) as pool:
        futures = {
            pool.submit(
                source.search,
                **{
                    'origin': origin,
                    'destination': destination,
                    'outbound_date': date,
                    'adults': adults,
                    'cabin_class': cabin,
                    **({'parallel': False} if isinstance(source, AggregatedFlightSource) else {}),
                },
            ): (origin, destination)
            for origin, destination in pairs
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                results[pair] = future.result() or []
            except NoResultsError:
                results[pair] = []
            except TravelAgentError as exc:
                errors.append(exc)
                logger.info('open_jaw: source failed for %s->%s: %s', *pair, exc.__class__.__name__)
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)
                logger.warning('open_jaw: unexpected failure for %s->%s: %s', *pair, exc.__class__.__name__)
    return results, errors


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------

@tool(args_schema=OpenJawSchema)
def open_jaw_search(params: OpenJawInput) -> dict[str, Any]:
    """Search open-jaw / flexible-destination round-trip itineraries.

    Returns a structured payload with ``status``, the expanded candidate
    airports, the resolved transit blacklist, a list of ranked
    combinations, and a Markdown summary the LLM can emit directly.
    """
    missing = [
        name for name in ('origin', 'destination_region', 'outbound_date', 'return_date')
        if not getattr(params, name)
    ]
    if missing:
        return degrade(MissingParameterError(missing))

    destinations = _resolve_destinations(params.destination_region)
    if not destinations:
        # The parameter is *present* but unrecognised, so use Ambiguous
        # rather than Missing — callers can distinguish "user forgot to
        # say where" from "the region name didn't map to any cities".
        return degrade(
            InvalidParameterError(
                'destination_region',
                params.destination_region,
                reason='could not be expanded into known regions or IATA airport codes',
            )
        )

    banned = _resolve_banned(params.avoid_transit)

    try:
        source = get_default_aggregator()
    except Exception as exc:  # pylint: disable=broad-except
        return degrade(exc)

    cabin = (params.cabin_class or 'economy').lower()
    adults = max(1, params.adults or 1)

    outbound_pairs = [(params.origin, dest) for dest in destinations]
    return_pairs = [(dest, params.origin) for dest in destinations]

    try:
        outbound_raw, outbound_errors = _fan_out(
            source, outbound_pairs, date=params.outbound_date, adults=adults, cabin=cabin,
        )
        return_raw, return_errors = _fan_out(
            source, return_pairs, date=params.return_date, adults=adults, cabin=cabin,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return degrade(exc)

    if (outbound_pairs and not any(outbound_raw.values()) and outbound_errors):
        return degrade(outbound_errors[0])
    if (return_pairs and not any(return_raw.values()) and return_errors):
        return degrade(return_errors[0])

    outbound_by_city = {
        dest: outbound_raw.get((params.origin, dest), [])
        for dest in destinations
    }
    return_by_city = {
        dest: return_raw.get((dest, params.origin), [])
        for dest in destinations
    }

    combinations = rank_open_jaw_combinations(
        outbound_by_city,
        return_by_city,
        banned_transit=banned,
        same_city=bool(params.same_city),
        top_n=params.top_n or 10,
        max_price=params.max_price,
    )

    if not combinations:
        return {
            'status': 'no_results',
            'origin': params.origin,
            'candidates': destinations,
            'banned_transit': sorted(banned),
            'message': (
                'No itineraries matched your constraints. Try relaxing the '
                'date range, max_price, or the transit blacklist.'
            ),
        }

    return {
        'status': 'ok',
        'origin': params.origin,
        'outbound_date': params.outbound_date,
        'return_date': params.return_date,
        'candidates': destinations,
        'banned_transit': sorted(banned),
        'same_city': bool(params.same_city),
        'count': len(combinations),
        'combinations': combinations,
        'summary_markdown': format_open_jaw_combinations(combinations, limit=params.top_n or 5),
    }
