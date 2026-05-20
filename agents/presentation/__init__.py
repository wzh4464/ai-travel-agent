"""Result processing and display helpers."""

from agents.presentation.formatter import (
    format_card,
    format_comparison_table,
    format_flight_list,
)
from agents.presentation.itinerary import (
    format_open_jaw_combinations,
    rank_open_jaw_combinations,
)
from agents.presentation.sorting import filter_flights, sort_flights

# ``_touches_banned_transit`` lives in :mod:`agents.presentation.sorting`
# and is a private helper. Internal callers import it directly from there.
__all__ = [
    'filter_flights',
    'sort_flights',
    'format_card',
    'format_comparison_table',
    'format_flight_list',
    'format_open_jaw_combinations',
    'rank_open_jaw_combinations',
]
