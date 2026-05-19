"""Unified flight schema and provider-specific normalizers.

Every data source is expected to produce :class:`Flight` instances (as plain
dicts so the LLM can read them). Keeping a single internal shape means the
sorting, filtering, and comparison tools never depend on upstream quirks.
"""

from __future__ import annotations

import datetime
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


# Anchored at both ends so we don't silently accept (and mis-parse) values
# like ``PT1H30M45S`` as 90 minutes. Days are honoured for multi-day
# itineraries (P1DT2H30M); seconds are tolerated but rounded to minutes.
_ISO_DURATION = re.compile(
    r'^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$',
    re.IGNORECASE,
)


def _parse_iso_duration(value: str) -> int:
    """Parse an ISO 8601 duration like ``PT10H30M`` or ``P1DT2H`` into minutes."""
    if not value:
        return 0
    m = _ISO_DURATION.match(value.strip())
    if not m:
        return 0
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    mins = int(m.group(3) or 0)
    secs = int(m.group(4) or 0)
    return days * 24 * 60 + hours * 60 + mins + secs // 60


def _minutes_between(start: str, end: str) -> int:
    """Best-effort minute diff between two ISO timestamps."""
    if not start or not end:
        return 0
    try:
        a = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
        b = datetime.datetime.fromisoformat(end.replace('Z', '+00:00'))
    except ValueError:
        return 0
    return max(0, int((b - a).total_seconds() // 60))


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
        booking_url=raw.get('booking_token', '') or 'https://www.google.com/flights',
        provider=provider,
        raw=raw,
    )


def normalize_amadeus(
    offer: dict,
    carriers: dict[str, str] | None = None,
    provider: str = 'amadeus',
) -> Flight:
    """Convert one Amadeus ``flight-offers`` entry into a canonical Flight.

    ``carriers`` is the ``dictionaries.carriers`` map from the Amadeus
    response — it translates a 2-letter carrier code into a display name.
    """
    carriers = carriers or {}
    itineraries = offer.get('itineraries', []) or []
    # Flatten segments across every itinerary so round-trip return legs are
    # not silently dropped. Track per-itinerary stop counts for the worst-case
    # ``stops`` figure (matches normalize_duffel's behaviour).
    segments: list[dict] = []
    per_itin_stops: list[int] = []
    for itin in itineraries:
        itin_segments = itin.get('segments', []) or []
        segments.extend(itin_segments)
        per_itin_stops.append(max(0, len(itin_segments) - 1))

    # Cabin class lives on travelerPricings[0].fareDetailsBySegment[i].cabin
    fare_details = []
    tps = offer.get('travelerPricings') or []
    if tps:
        fare_details = tps[0].get('fareDetailsBySegment', []) or []

    legs: list[FlightLeg] = []
    for i, seg in enumerate(segments):
        carrier_code = seg.get('carrierCode', '') or ''
        cabin = ''
        if i < len(fare_details):
            cabin = (fare_details[i].get('cabin') or '').lower()
        legs.append(
            FlightLeg(
                airline=carriers.get(carrier_code, carrier_code),
                flight_number=f'{carrier_code}{seg.get("number", "")}',
                departure_airport=(seg.get('departure') or {}).get('iataCode', ''),
                departure_time=(seg.get('departure') or {}).get('at', ''),
                arrival_airport=(seg.get('arrival') or {}).get('iataCode', ''),
                arrival_time=(seg.get('arrival') or {}).get('at', ''),
                duration_minutes=_parse_iso_duration(seg.get('duration', '')),
                aircraft=(seg.get('aircraft') or {}).get('code', ''),
                cabin_class=cabin,
            )
        )

    price_block = offer.get('price') or {}
    total_duration = sum(
        _parse_iso_duration((it.get('duration', '') or '')) for it in itineraries
    )
    return Flight(
        flight_id=str(offer.get('id') or _stable_id(offer)),
        price=float(price_block.get('total', 0) or 0),
        currency=price_block.get('currency', 'USD') or 'USD',
        total_duration_minutes=total_duration,
        stops=max(per_itin_stops) if per_itin_stops else 0,
        legs=legs,
        airline_logo='',
        booking_url='',  # Amadeus Self-Service has no direct deep link; use booking API
        provider=provider,
        raw=offer,
    )


def normalize_kiwi(offer: dict, provider: str = 'kiwi') -> Flight:
    """Convert one Kiwi Tequila ``/v2/search`` item into a canonical Flight."""
    route = offer.get('route', []) or []
    legs: list[FlightLeg] = []
    for seg in route:
        airline = seg.get('airline', '') or ''
        dep = seg.get('local_departure') or seg.get('utc_departure') or ''
        arr = seg.get('local_arrival') or seg.get('utc_arrival') or ''
        legs.append(
            FlightLeg(
                airline=airline,
                flight_number=f'{airline}{seg.get("flight_no", "")}',
                departure_airport=seg.get('flyFrom', '') or '',
                departure_time=dep,
                arrival_airport=seg.get('flyTo', '') or '',
                arrival_time=arr,
                duration_minutes=_minutes_between(dep, arr),
                aircraft=seg.get('equipment', '') or '',
                cabin_class='',
            )
        )

    duration_block = offer.get('duration') or {}
    total_seconds = duration_block.get('total') or 0
    try:
        total_minutes = int(total_seconds) // 60
    except (TypeError, ValueError):
        total_minutes = 0

    return Flight(
        flight_id=str(offer.get('id') or _stable_id(offer)),
        price=float(offer.get('price', 0) or 0),
        currency='USD',
        total_duration_minutes=total_minutes,
        stops=max(0, len(legs) - 1),
        legs=legs,
        airline_logo='',
        booking_url=offer.get('deep_link', '') or '',
        provider=provider,
        raw=offer,
    )
