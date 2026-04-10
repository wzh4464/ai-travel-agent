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
from agents.presentation.sorting import (
    _touches_banned_transit,
    filter_flights,
    sort_flights,
)

__all__ = [
    'filter_flights',
    'sort_flights',
    '_touches_banned_transit',
    'format_card',
    'format_comparison_table',
    'format_flight_list',
    'format_open_jaw_combinations',
    'rank_open_jaw_combinations',
]
