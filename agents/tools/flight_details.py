"""Tool that extracts detailed information from a Flight payload.

Google Flights (via SerpAPI) does not expose a per-id lookup endpoint, but
the search response already contains baggage allowance, layover details, and
booking URLs. This tool parses that data into a structured shape that the
LLM can render to the user.
"""

from __future__ import annotations

from typing import Any, List, Optional

from agents._pydantic_compat import BaseModel, Field
from langchain_core.tools import tool


class FlightDetailsInput(BaseModel):
    flight_id: Optional[str] = Field(
        default=None,
        description='The flight_id returned by flights_finder to look up.',
    )
    flights: List[dict] = Field(
        description='The list of Flight dicts previously returned by flights_finder.',
    )


class FlightDetailsSchema(BaseModel):
    params: FlightDetailsInput


def _find_flight(flights: list[dict], flight_id: str | None) -> dict | None:
    if not flights:
        return None
    if not flight_id:
        return flights[0]
    for f in flights:
        if f.get('flight_id') == flight_id:
            return f
    return None


def _layovers(raw: dict) -> list[dict[str, Any]]:
    out = []
    for lo in raw.get('layovers', []) or []:
        out.append(
            {
                'airport': lo.get('id', ''),
                'name': lo.get('name', ''),
                'duration_minutes': lo.get('duration', 0),
                'overnight': bool(lo.get('overnight', False)),
            }
        )
    return out


@tool(args_schema=FlightDetailsSchema)
def get_flight_details(params: FlightDetailsInput) -> dict:
    """Return structured details (layovers, baggage, booking URL) for a flight.

    Pass the full list of flights from flights_finder together with the
    specific flight_id you want to inspect.
    """
    flight = _find_flight(params.flights, params.flight_id)
    if flight is None:
        return {
            'status': 'not_found',
            'message': 'No flight matched the provided flight_id.',
        }
    raw = flight.get('raw', {}) or {}
    extensions = raw.get('extensions', []) or []
    return {
        'status': 'ok',
        'flight_id': flight.get('flight_id'),
        'price': flight.get('price'),
        'currency': flight.get('currency', 'USD'),
        'total_duration_minutes': flight.get('total_duration_minutes'),
        'stops': flight.get('stops'),
        'legs': flight.get('legs', []),
        'layovers': _layovers(raw),
        'baggage': [e for e in extensions if 'bag' in str(e).lower()],
        'amenities': [e for e in extensions if 'bag' not in str(e).lower()],
        'carbon_emissions': raw.get('carbon_emissions', {}),
        'booking_url': flight.get('booking_url'),
        'airline_logo': flight.get('airline_logo'),
    }
