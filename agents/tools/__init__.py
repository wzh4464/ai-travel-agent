"""LangChain tool definitions exposed to the agent."""

from agents.tools.airport_code import get_airport_code, lookup as lookup_airport_code
from agents.tools.compare_prices import compare_prices
from agents.tools.flight_details import get_flight_details
from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder
from agents.tools.seat_availability import check_seat_availability

FLIGHT_TOOLS = [
    flights_finder,
    get_flight_details,
    compare_prices,
    check_seat_availability,
    get_airport_code,
]

ALL_TOOLS = FLIGHT_TOOLS + [hotels_finder]

__all__ = [
    'flights_finder',
    'get_flight_details',
    'compare_prices',
    'check_seat_availability',
    'get_airport_code',
    'hotels_finder',
    'lookup_airport_code',
    'FLIGHT_TOOLS',
    'ALL_TOOLS',
]
