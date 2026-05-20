"""Real-time seat availability lookup.

Google Flights does not publish a seat map, so the current implementation
uses the cabin-class and passenger-count information from the search result
to return a best-effort estimate. The tool is designed to be swapped for a
direct airline API (e.g. Amadeus Seat Map Display) without changing its
signature.
"""

from __future__ import annotations

from typing import List, Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool


class SeatAvailabilityInput(BaseModel):
    flight_id: str = Field(description='Flight identifier returned by flights_finder.')
    flights: List[dict] = Field(description='List of Flight dicts from flights_finder.')
    requested_seats: Optional[int] = Field(1, description='Number of seats the user wants.')


class SeatAvailabilitySchema(BaseModel):
    params: SeatAvailabilityInput


@tool(args_schema=SeatAvailabilitySchema)
def check_seat_availability(params: SeatAvailabilityInput) -> dict:
    """Check whether the requested number of seats is likely available."""
    match = next((f for f in params.flights if f.get('flight_id') == params.flight_id), None)
    if match is None:
        return {
            'status': 'not_found',
            'message': 'Unknown flight_id. Run flights_finder again to refresh the list.',
        }

    raw = match.get('raw', {}) or {}
    cabin = next(
        (leg.get('cabin_class') or 'economy' for leg in match.get('legs', []) or [] if leg.get('cabin_class')),
        'economy',
    )
    seats_left = raw.get('seats_left')
    if seats_left is None:
        estimate = 'likely_available' if (params.requested_seats or 1) <= 4 else 'call_airline'
        return {
            'status': 'estimated',
            'flight_id': params.flight_id,
            'cabin_class': cabin,
            'requested_seats': params.requested_seats or 1,
            'availability': estimate,
            'message': (
                'The provider does not expose a real-time seat map. Proceed to '
                'booking to confirm actual availability.'
            ),
        }
    return {
        'status': 'ok',
        'flight_id': params.flight_id,
        'cabin_class': cabin,
        'seats_left': int(seats_left),
        'sufficient': int(seats_left) >= (params.requested_seats or 1),
    }
