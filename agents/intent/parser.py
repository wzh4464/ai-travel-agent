"""Structured intent extraction and multi-turn dialog state."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from agents.intent.fuzzy import (
    interpret_fuzzy_date,
    interpret_price_preference,
    interpret_stops_preference,
)
from agents.intent.iata import lookup as lookup_airport_code


@dataclass
class TravelIntent:
    """Structured form of a flight-search request."""

    origin_city: Optional[str] = None
    origin_code: Optional[str] = None
    destination_city: Optional[str] = None
    destination_code: Optional[str] = None
    outbound_date: Optional[str] = None
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    cabin_class: str = 'economy'
    max_stops: Optional[int] = None
    max_price: Optional[float] = None
    sort_by: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DialogState:
    """Tracks accumulated intent across turns in a single conversation thread."""

    intent: TravelIntent = field(default_factory=TravelIntent)
    clarifications_asked: list[str] = field(default_factory=list)

    def merge(self, new: TravelIntent) -> None:
        """Fill in any fields on the tracked intent that the new turn resolved."""
        for key, value in new.as_dict().items():
            if value in (None, 0, '', 'economy'):
                continue
            # 0 is a valid value for max_stops (non-stop), handle explicitly
            current = getattr(self.intent, key)
            if current in (None, 0, '', 'economy') or key == 'max_stops':
                setattr(self.intent, key, value)


REQUIRED_SLOTS = ('origin_code', 'destination_code', 'outbound_date')


def missing_slots(intent: TravelIntent) -> list[str]:
    return [slot for slot in REQUIRED_SLOTS if not getattr(intent, slot)]


def clarification_question(slot: str) -> str:
    prompts = {
        'origin_code': 'Which city will you be flying from?',
        'destination_code': 'Which city would you like to fly to?',
        'outbound_date': 'What date do you want to depart? (e.g. 2026-05-01 or "next weekend")',
    }
    return prompts.get(slot, f'Could you clarify: {slot}?')


# ---------------------------------------------------------------------------
# Heuristic extraction
# ---------------------------------------------------------------------------

_FROM_TO_EN = re.compile(r'from\s+([\w\s]+?)\s+to\s+([\w\s]+?)(?:\s+on|\s+in|\s+next|\s+this|\s+for|[,.?!]|$)', re.I)
_FROM_TO_CN = re.compile(r'从\s*([\u4e00-\u9fa5A-Za-z ]+?)\s*(?:到|飞)\s*([\u4e00-\u9fa5A-Za-z ]+?)(?:\s|，|。|$)')
_ADULTS_EN = re.compile(r'(\d+)\s+(?:adult|passenger|people|pax)', re.I)
_ADULTS_CN = re.compile(r'(\d+)\s*(?:人|名乘客|位)')
_CABIN_WORDS = {
    'business': ('business', '商务'),
    'first': ('first class', '头等舱', 'first-class'),
    'premium_economy': ('premium economy', '超级经济', '豪华经济'),
    'economy': ('economy', '经济舱'),
}


def _extract_pair(text: str) -> tuple[Optional[str], Optional[str]]:
    for pattern in (_FROM_TO_EN, _FROM_TO_CN):
        m = pattern.search(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None, None


def _extract_cabin(text: str) -> str:
    s = text.lower()
    for canonical, markers in _CABIN_WORDS.items():
        if any(m in s for m in markers):
            return canonical
    return 'economy'


def _extract_pax(text: str) -> int:
    for pattern in (_ADULTS_EN, _ADULTS_CN):
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    return 1


def extract_intent(text: str) -> TravelIntent:
    """Heuristic extractor that runs *before* the LLM.

    The LLM still has the final say — this is just a fast, deterministic
    first pass that populates whatever slots we can be confident about.
    """
    intent = TravelIntent()
    if not text:
        return intent

    origin, destination = _extract_pair(text)
    if origin:
        codes = lookup_airport_code(origin)
        intent.origin_city = origin
        if codes:
            intent.origin_code = codes[0]
    if destination:
        codes = lookup_airport_code(destination)
        intent.destination_city = destination
        if codes:
            intent.destination_code = codes[0]

    fuzzy = interpret_fuzzy_date(text)
    if fuzzy:
        intent.outbound_date = fuzzy.start
        intent.return_date = fuzzy.end

    price_pref = interpret_price_preference(text)
    if price_pref:
        intent.max_price = price_pref.get('max_price')
        intent.sort_by = price_pref.get('sort_by')

    stops = interpret_stops_preference(text)
    if stops is not None:
        intent.max_stops = stops

    intent.cabin_class = _extract_cabin(text)
    intent.adults = _extract_pax(text)
    return intent
