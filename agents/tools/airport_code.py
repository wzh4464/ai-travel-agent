"""City-to-IATA airport code lookup tool.

The underlying lookup table lives in :mod:`agents.intent.iata` so both the
intent parser and the LLM-facing tool share a single source of truth.
"""

from __future__ import annotations

from agents._pydantic_compat import BaseModel, Field
from langchain_core.tools import tool

from agents.intent.iata import CITY_TO_IATA, lookup

# Re-export so existing imports keep working.
__all__ = ['CITY_TO_IATA', 'lookup', 'get_airport_code', 'AirportCodeInput']


class AirportCodeInput(BaseModel):
    city_name: str = Field(description='City name in any common spelling, e.g. "New York", "东京".')


@tool
def get_airport_code(city_name: str) -> dict:
    """Resolve a city name to one or more IATA airport codes."""
    codes = lookup(city_name)
    if not codes:
        return {
            'status': 'not_found',
            'query': city_name,
            'message': (
                'No airport code was found for this city. Ask the user to '
                'provide the IATA code directly or try a nearby major city.'
            ),
        }
    return {
        'status': 'ok',
        'query': city_name,
        'primary': codes[0],
        'all': codes,
    }
