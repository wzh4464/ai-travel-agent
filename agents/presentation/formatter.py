"""Human-friendly views over canonical Flight dicts.

These helpers render flight data as Markdown strings that Streamlit can
display with ``st.markdown``. They complement the LLM's free-form output
by providing deterministic card / table / comparison views.
"""

from __future__ import annotations


def _fmt_duration(minutes: int) -> str:
    minutes = int(minutes or 0)
    hours, mins = divmod(minutes, 60)
    return f'{hours}h {mins}m' if hours else f'{mins}m'


def _price(flight: dict) -> str:
    currency = flight.get('currency', 'USD')
    return f'{currency} {float(flight.get("price") or 0):.0f}'


def _primary_airline(flight: dict) -> str:
    legs = flight.get('legs') or []
    return legs[0].get('airline', 'Unknown') if legs else 'Unknown'


def format_card(flight: dict) -> str:
    """Render one flight as a Markdown card."""
    legs = flight.get('legs') or []
    if not legs:
        return f'- {_primary_airline(flight)} — {_price(flight)}'

    first, last = legs[0], legs[-1]
    header = f'### {_primary_airline(flight)} — {_price(flight)}'
    route = (
        f'{first.get("departure_airport", "???")} '
        f'{first.get("departure_time", "")} → '
        f'{last.get("arrival_airport", "???")} '
        f'{last.get("arrival_time", "")}'
    )
    stops = flight.get('stops', 0)
    stops_text = 'Non-stop' if stops == 0 else f'{stops} stop' + ('s' if stops > 1 else '')
    duration = _fmt_duration(flight.get('total_duration_minutes', 0))

    lines = [
        header,
        f'**Route:** {route}',
        f'**Duration:** {duration}  •  **{stops_text}**',
    ]
    logo = flight.get('airline_logo')
    if logo:
        lines.append(f'![{_primary_airline(flight)}]({logo})')
    if flight.get('booking_url'):
        lines.append(f'[Book this flight]({flight["booking_url"]})')
    return '\n'.join(lines)


def format_flight_list(flights: list[dict], limit: int = 5) -> str:
    cards = [format_card(f) for f in flights[:limit]]
    return '\n\n---\n\n'.join(cards) if cards else '_No flights to display._'


def format_comparison_table(flights: list[dict], limit: int = 5) -> str:
    """Render a side-by-side comparison of the top results as a Markdown table."""
    if not flights:
        return '_No flights to compare._'
    header = '| # | Airline | Route | Stops | Duration | Price |'
    sep = '|---|---------|-------|-------|----------|-------|'
    rows = [header, sep]
    for idx, f in enumerate(flights[:limit], 1):
        legs = f.get('legs') or []
        first = legs[0] if legs else {}
        last = legs[-1] if legs else {}
        route = f'{first.get("departure_airport", "???")}→{last.get("arrival_airport", "???")}'
        stops = f.get('stops', 0)
        stops_text = 'Non-stop' if stops == 0 else f'{stops}'
        rows.append(
            f'| {idx} | {_primary_airline(f)} | {route} | {stops_text} | '
            f'{_fmt_duration(f.get("total_duration_minutes", 0))} | {_price(f)} |'
        )
    return '\n'.join(rows)
