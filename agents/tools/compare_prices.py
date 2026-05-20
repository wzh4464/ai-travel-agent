"""Sort / filter / compare tool for a list of previously-fetched flights."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from agents.presentation.sorting import filter_flights, sort_flights


class ComparePricesInput(BaseModel):
    flights: List[dict] = Field(description='List of Flight dicts returned by flights_finder.')
    sort_by: Optional[str] = Field(
        'price',
        description='Sort key: "price", "duration", "stops", or "departure".',
    )
    max_price: Optional[float] = Field(default=None, description='Optional max price filter.')
    max_stops: Optional[int] = Field(default=None, description='Optional maximum number of stops.')
    airlines: Optional[List[str]] = Field(
        default=None,
        description='Optional list of airline name substrings to include.',
    )
    limit: Optional[int] = Field(5, description='Maximum number of results to return.')


class ComparePricesSchema(BaseModel):
    params: ComparePricesInput


@tool(args_schema=ComparePricesSchema)
def compare_prices(params: ComparePricesInput) -> dict[str, Any]:
    """Sort and filter an existing list of flights.

    This tool never hits the network. It expects the ``flights`` payload
    returned by :func:`flights_finder` (or any other source that emits the
    canonical Flight shape).
    """
    filtered = filter_flights(
        params.flights,
        max_price=params.max_price,
        max_stops=params.max_stops,
        airlines=params.airlines,
    )
    ranked = sort_flights(filtered, key=params.sort_by or 'price')
    limit = params.limit or 5
    return {
        'status': 'ok',
        'count': len(ranked),
        'sort_by': params.sort_by or 'price',
        'filters_applied': {
            'max_price': params.max_price,
            'max_stops': params.max_stops,
            'airlines': params.airlines,
        },
        'results': ranked[:limit],
    }
