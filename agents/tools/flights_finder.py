"""Primary flight search tool.

Delegates the actual HTTP call to :mod:`agents.data_sources` so that retries,
rate limiting, and response normalisation happen in a single place.
"""

from __future__ import annotations

from typing import Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from agents.data_sources import get_default_source
from agents.errors import MissingParameterError, degrade


class FlightsInput(BaseModel):
    departure_airport: Optional[str] = Field(description='Departure airport code (IATA), e.g. "JFK".')
    arrival_airport: Optional[str] = Field(description='Arrival airport code (IATA), e.g. "LHR".')
    outbound_date: Optional[str] = Field(description='Outbound date in YYYY-MM-DD.')
    return_date: Optional[str] = Field(default=None, description='Optional return date in YYYY-MM-DD.')
    adults: Optional[int] = Field(1, description='Number of adult passengers.')
    children: Optional[int] = Field(0, description='Number of children.')
    infants_in_seat: Optional[int] = Field(0, description='Number of infants occupying a seat.')
    infants_on_lap: Optional[int] = Field(0, description='Number of infants on lap.')
    cabin_class: Optional[str] = Field(
        'economy',
        description='Cabin class: economy, premium_economy, business, or first.',
    )
    max_stops: Optional[int] = Field(
        default=None,
        description='Maximum allowed stops (0 = non-stop, 1 = <=1 stop, 2 = <=2 stops). Omit for any.',
    )


class FlightsInputSchema(BaseModel):
    params: FlightsInput


@tool(args_schema=FlightsInputSchema)
def flights_finder(params: FlightsInput):
    """Search flights across configured data sources and return normalised results.

    Returns a list of Flight dicts (see agents.data_sources.normalizer.Flight)
    on success, or a structured error payload on failure.
    """
    missing = [
        name
        for name in ('departure_airport', 'arrival_airport', 'outbound_date')
        if not getattr(params, name)
    ]
    if missing:
        return degrade(MissingParameterError(missing))

    try:
        source = get_default_source()
        return source.search(
            origin=params.departure_airport,
            destination=params.arrival_airport,
            outbound_date=params.outbound_date,
            return_date=params.return_date,
            adults=params.adults or 1,
            children=params.children or 0,
            infants_in_seat=params.infants_in_seat or 0,
            infants_on_lap=params.infants_on_lap or 0,
            cabin_class=params.cabin_class or 'economy',
            max_stops=params.max_stops,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return degrade(exc)
