"""Open-jaw combination ranking and rendering.

The open-jaw search tool ultimately needs to answer "what are the
cheapest outbound+return combinations, given a basket of candidate
cities"? The pure-data part of that lives here so it can be unit tested
without a live aggregator.
"""

from __future__ import annotations

from typing import Any, Mapping

from agents.presentation.sorting import _touches_banned_transit


def rank_open_jaw_combinations(
    outbound_by_city: Mapping[str, list[dict]],
    return_by_city: Mapping[str, list[dict]],
    *,
    banned_transit: set[str] | None = None,
    same_city: bool = False,
    top_n: int = 10,
    max_price: float | None = None,
    per_city_candidates: int = 3,
) -> list[dict[str, Any]]:
    """Combine outbound and return flights into ranked itinerary options.

    Parameters
    ----------
    outbound_by_city:
        Mapping from the "entry" IATA code to a list of candidate
        outbound flights landing there.
    return_by_city:
        Mapping from the "exit" IATA code to a list of candidate return
        flights departing from there.
    banned_transit:
        Intermediate airports to reject. Applied to both directions
        individually before combination.
    same_city:
        When True, the exit city must equal the entry city (traditional
        round-trip). When False, any pairing is allowed (open-jaw).
    top_n:
        Maximum number of combinations to return.
    max_price:
        Optional cap on the total combined price.
    per_city_candidates:
        To keep the cartesian explosion manageable, only the cheapest
        ``per_city_candidates`` flights per entry/exit city are used.
    """
    banned = banned_transit or set()

    def _prep(flights: list[dict]) -> list[dict]:
        cleaned = [f for f in flights if not _touches_banned_transit(f, banned)] if banned else list(flights)
        cleaned.sort(key=lambda f: float(f.get('price') or 10**9))
        return cleaned[:per_city_candidates]

    prepped_out = {city: _prep(flights) for city, flights in outbound_by_city.items()}
    prepped_ret = {city: _prep(flights) for city, flights in return_by_city.items()}

    combinations: list[dict[str, Any]] = []
    for entry_city, outbounds in prepped_out.items():
        if not outbounds:
            continue
        for ob in outbounds:
            if same_city:
                exits = [(entry_city, ret) for ret in prepped_ret.get(entry_city, [])]
            else:
                exits = [
                    (exit_city, ret)
                    for exit_city, rets in prepped_ret.items()
                    for ret in rets
                ]
            for exit_city, ret in exits:
                total = float(ob.get('price') or 0) + float(ret.get('price') or 0)
                if max_price is not None and total > max_price:
                    continue
                combinations.append({
                    'total_price': round(total, 2),
                    'currency': ob.get('currency') or ret.get('currency') or 'USD',
                    'entry_city': entry_city,
                    'exit_city': exit_city,
                    'open_jaw': entry_city != exit_city,
                    'outbound': ob,
                    'return': ret,
                })

    combinations.sort(key=lambda c: c['total_price'])
    return combinations[:top_n]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt_duration(minutes: int) -> str:
    minutes = int(minutes or 0)
    h, m = divmod(minutes, 60)
    return f'{h}h {m}m' if h else f'{m}m'


def _compact_flight_line(flight: dict) -> str:
    legs = flight.get('legs') or []
    if not legs:
        return 'Unknown'
    first, last = legs[0], legs[-1]
    stops = int(flight.get('stops') or 0)
    if stops == 0:
        stops_text = 'non-stop'
    else:
        stops_text = f'{stops} stop' + ('s' if stops > 1 else '')
        transit_airports = [leg.get('arrival_airport', '') for leg in legs[:-1]]
        if transit_airports:
            stops_text += f' via {"/".join(transit_airports)}'
    airline = first.get('airline', '')
    currency = flight.get('currency', '')
    price = float(flight.get('price') or 0)
    return (
        f'{airline} {first.get("departure_airport")} → {last.get("arrival_airport")}, '
        f'{stops_text}, {_fmt_duration(flight.get("total_duration_minutes", 0))}, '
        f'{currency} {price:,.0f}'
    )


def format_open_jaw_combinations(combinations: list[dict], *, limit: int = 5) -> str:
    """Render the top ``limit`` open-jaw combinations as a Markdown block."""
    if not combinations:
        return '_No itineraries matched your constraints._'
    lines: list[str] = []
    for i, combo in enumerate(combinations[:limit], 1):
        header = (
            f'### Option {i}: '
            f'{combo["entry_city"]} → {combo["exit_city"]}'
        )
        if combo['open_jaw']:
            header += ' *(open-jaw)*'
        header += f' — **{combo["currency"]} {combo["total_price"]:,.0f} total**'
        lines.append(header)
        lines.append(f'- **Outbound:** {_compact_flight_line(combo["outbound"])}')
        lines.append(f'- **Return:** {_compact_flight_line(combo["return"])}')
        lines.append('')
    return '\n'.join(lines).rstrip()
