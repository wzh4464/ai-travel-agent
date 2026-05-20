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
    # adults / cabin_class are Optional so the dialog can distinguish "user
    # didn't say" from "user said 1 / economy". Tool-call construction layers
    # apply the actual booking defaults via ``intent.adults or 1`` etc.
    adults: Optional[int] = None
    children: Optional[int] = None
    cabin_class: Optional[str] = None
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
        """Overlay slots resolved by ``new`` onto the tracked intent.

        A non-sentinel value from ``new`` always replaces the tracked slot
        so the user can correct themselves — "actually from SFO" after an
        earlier "from LAX", "actually economy" after "business", or
        "actually 1 adult" after "3 adults".

        Sentinel / unset values are skipped so a later turn that omits a
        date does not clobber an earlier one. ``max_stops=0`` is a real
        preference (non-stop), so its only sentinel is ``None``; every
        other field treats ``None`` and ``''`` as unset.
        """
        for key, value in new.as_dict().items():
            if key == 'max_stops':
                if value is None:
                    continue
            elif value in (None, ''):
                continue
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

# Terminator words that should *end* a city-name capture. Without these in
# the boundary group, "to Tokyo direct on 2026-05-01" would greedily capture
# "Tokyo direct" as the destination.
_TERMINATORS = (
    r'on|in|next|this|for|direct|non-?stop|nonstop|cheap|cheapest|'
    r'business|economy|first|premium|with|via|by'
)
# City body: Unicode-friendly so "São Paulo" / "München" survive intact.
# One leading letter/digit + any mix of letters, digits, spaces, hyphens,
# apostrophes. Lazy so the terminator group can fire.
_CITY_CHARS = r"[^\W_][\w '\-]*?"
_FROM_TO_EN = re.compile(
    rf'from\s+({_CITY_CHARS})\s+to\s+({_CITY_CHARS})'
    rf'(?:\s+(?:{_TERMINATORS})\b|[,.?!]|$)',
    re.I | re.U,
)
# Origin-only follow-ups: "from SFO", "leaving from Beijing tomorrow".
_FROM_ONLY_EN = re.compile(
    rf'(?:^|\s)from\s+({_CITY_CHARS})'
    rf'(?:\s+(?:to|{_TERMINATORS})\b|[,.?!]|$)',
    re.I | re.U,
)
# Destination-only follow-ups: "fly to Tokyo", "heading to LHR".
_TO_ONLY_EN = re.compile(
    rf'(?:^|\s)(?:fly|go|travel|head(?:ing)?)\s+to\s+({_CITY_CHARS})'
    rf'(?:\s+(?:{_TERMINATORS})\b|[,.?!]|$)',
    re.I | re.U,
)
# CJK terminators mirror _TERMINATORS: words that should *end* a city-name
# capture so we don't glue date / cabin / preference markers onto the
# destination ("从北京到东京下周一" must capture "东京", not "东京下周一").
# Includes the origin-only verbs (出发/动身/启程/起飞) so the from→to regex
# stops cleanly before "从X出发" — and the single-side _FROM_ONLY_CN below
# can capture an origin without those verbs polluting it.
_CN_CITY_TERMINATORS = (
    r'下周|本周|这周|明天|后天|今天|早上|中午|晚上|'
    r'直飞|直达|经济|商务|头等|便宜|最便宜|往返|单程|'
    r'含|带|从|不要|不经|避免|月|号|日|'
    r'出发|动身|启程|起飞'
)
_FROM_TO_CN = re.compile(
    r'从\s*([\u4e00-\u9fa5A-Za-z ]+?)\s*(?:到|飞)\s*'
    r'([\u4e00-\u9fa5A-Za-z ]+?)'
    rf'(?=\s|，|。|,|\.|$|{_CN_CITY_TERMINATORS}|\d)'
)
# Origin-only CJK: "从香港出发", "从北京动身".
_FROM_ONLY_CN = re.compile(
    r'从\s*([\u4e00-\u9fa5A-Za-z ]+?)\s*(?:出发|动身|启程|起飞)'
)
_ADULTS_EN = re.compile(r'(\d+)\s+(?:adult|passenger|people|pax)', re.I)
_ADULTS_CN = re.compile(r'(\d+)\s*(?:人|名乘客|位)')
_CABIN_WORDS = {
    'business': ('business', '商务'),
    'first': ('first class', '头等舱', 'first-class'),
    'premium_economy': ('premium economy', '超级经济', '豪华经济'),
    'economy': ('economy', '经济舱'),
}


def _extract_pair(text: str) -> tuple[Optional[str], Optional[str]]:
    # Try the full from→to pair first (either language).
    for pattern in (_FROM_TO_EN, _FROM_TO_CN):
        m = pattern.search(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    # Fall back to single-side captures. Either or both may resolve in a
    # single turn ("fly to Tokyo" / "from Beijing next weekend").
    origin: Optional[str] = None
    destination: Optional[str] = None
    m = _FROM_ONLY_EN.search(text) or _FROM_ONLY_CN.search(text)
    if m:
        origin = m.group(1).strip()
    m = _TO_ONLY_EN.search(text)
    if m:
        destination = m.group(1).strip()
    return origin, destination


def _extract_cabin(text: str) -> Optional[str]:
    """Return the canonical cabin class the user mentioned, or ``None``.

    Returning ``None`` when no cabin word matched is what lets
    :class:`DialogState.merge` avoid clobbering an earlier 'business'
    choice with the dataclass default 'economy' on every subsequent turn.
    """
    s = text.lower()
    for canonical, markers in _CABIN_WORDS.items():
        if any(m in s for m in markers):
            return canonical
    return None


def _extract_pax(text: str) -> Optional[int]:
    """Return the explicit passenger count the user gave, or ``None``.

    Mirrors :func:`_extract_cabin`: ``None`` means "user didn't say". The
    dataclass default of ``1`` only applies when no turn ever mentions a
    count.
    """
    for pattern in (_ADULTS_EN, _ADULTS_CN):
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    return None


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

    cabin = _extract_cabin(text)
    if cabin is not None:
        intent.cabin_class = cabin
    pax = _extract_pax(text)
    if pax is not None:
        intent.adults = pax
    return intent
