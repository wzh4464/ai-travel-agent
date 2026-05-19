"""Pure-Python sort / filter operations on canonical Flight dicts."""

from __future__ import annotations

from typing import Iterable, Optional


def _first_departure(flight: dict) -> str:
    legs = flight.get('legs') or []
    return legs[0].get('departure_time', '') if legs else ''


_SORT_KEYS = {
    'price': lambda f: (float(f.get('price') or 0) or float('inf')),
    'duration': lambda f: (int(f.get('total_duration_minutes') or 0) or 10**9),
    'stops': lambda f: (int(f.get('stops') or 0), float(f.get('price') or 0)),
    'departure': lambda f: _first_departure(f),
}


def sort_flights(flights: list[dict], key: str = 'price') -> list[dict]:
    sort_fn = _SORT_KEYS.get(key, _SORT_KEYS['price'])
    return sorted(flights, key=sort_fn)


def _airline_matches(flight: dict, airlines: Iterable[str]) -> bool:
    wanted = {a.lower() for a in airlines}
    for leg in flight.get('legs', []) or []:
        name = (leg.get('airline') or '').lower()
        if any(w in name for w in wanted):
            return True
    return False


def _touches_banned_transit(flight: dict, banned: set[str]) -> bool:
    """Return True if ``flight`` routes through any airport in ``banned``.

    Only *intermediate* airports count as transits. The first leg's
    departure is the user's origin and the last leg's arrival is the
    chosen destination, neither of which is a transit.
    """
    legs = flight.get('legs') or []
    if len(legs) <= 1:
        return False  # non-stop flights have no transit
    for leg in legs[:-1]:
        if leg.get('arrival_airport') in banned:
            return True
    return False


def filter_flights(
    flights: list[dict],
    *,
    max_price: Optional[float] = None,
    max_stops: Optional[int] = None,
    airlines: Optional[list[str]] = None,
    avoid_transit: Optional[set[str]] = None,
) -> list[dict]:
    out: list[dict] = []
    for f in flights:
        if max_price is not None and float(f.get('price') or 0) > max_price:
            continue
        if max_stops is not None and int(f.get('stops') or 0) > max_stops:
            continue
        if airlines and not _airline_matches(f, airlines):
            continue
        if avoid_transit and _touches_banned_transit(f, avoid_transit):
            continue
        out.append(f)
    return out
