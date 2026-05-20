"""Unified flight schema and provider-specific normalizers.

Every data source is expected to produce :class:`Flight` instances (as plain
dicts so the LLM can read them). Keeping a single internal shape means the
sorting, filtering, and comparison tools never depend on upstream quirks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FlightLeg:
    airline: str
    flight_number: str
    departure_airport: str
    departure_time: str
    arrival_airport: str
    arrival_time: str
    duration_minutes: int
    aircraft: str = ''
    cabin_class: str = ''


@dataclass
class Flight:
    """Canonical flight shape used across all tools."""

    flight_id: str
    price: float
    currency: str
    total_duration_minutes: int
    stops: int
    legs: list[FlightLeg] = field(default_factory=list)
    airline_logo: str = ''
    booking_url: str = ''
    provider: str = ''
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_id(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()[:12]


def _minutes(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_PRICE_NUMERIC = re.compile(r'[^0-9.]+')


def _coerce_price(raw_price) -> float:
    """SerpAPI may return prices as ``"$702"`` or ``702`` — accept both."""
    if raw_price in (None, ''):
        return 0.0
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    cleaned = _PRICE_NUMERIC.sub('', str(raw_price))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def normalize_serpapi(raw: dict, provider: str = 'serpapi-google-flights') -> Flight:
    """Convert one SerpAPI google_flights ``best_flights`` item into a Flight."""
    legs_raw = raw.get('flights', []) or []
    legs: list[FlightLeg] = []
    for leg in legs_raw:
        legs.append(
            FlightLeg(
                airline=leg.get('airline', ''),
                flight_number=leg.get('flight_number', ''),
                departure_airport=(leg.get('departure_airport') or {}).get('id', ''),
                departure_time=(leg.get('departure_airport') or {}).get('time', ''),
                arrival_airport=(leg.get('arrival_airport') or {}).get('id', ''),
                arrival_time=(leg.get('arrival_airport') or {}).get('time', ''),
                duration_minutes=_minutes(leg.get('duration', 0)),
                aircraft=leg.get('airplane', ''),
                cabin_class=leg.get('travel_class', ''),
            )
        )
    stops = max(0, len(legs) - 1)
    return Flight(
        flight_id=_stable_id(raw),
        price=_coerce_price(raw.get('price')),
        currency='USD',
        total_duration_minutes=_minutes(raw.get('total_duration', 0)),
        stops=stops,
        legs=legs,
        airline_logo=raw.get('airline_logo', ''),
        # booking_token is an opaque SerpAPI id, not a clickable URL. Use
        # the Google Flights homepage so downstream Markdown renderers
        # don't produce a broken link.
        booking_url='https://www.google.com/flights',
        provider=provider,
        raw=raw,
    )
