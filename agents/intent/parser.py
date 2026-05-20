"""Structured intent extraction and multi-turn dialog state."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

from agents.intent.fuzzy import (
    interpret_fuzzy_date,
    interpret_price_preference,
    interpret_stops_preference,
)
from agents.intent.iata import lookup as lookup_airport_code
from agents.regions import (
    REGION_CJK_ALIASES,
    TRANSIT_BLACKLISTS,
)


@dataclass
class TravelIntent:
    """Structured form of a flight-search request."""

    origin_city: Optional[str] = None
    origin_code: Optional[str] = None
    destination_city: Optional[str] = None
    destination_code: Optional[str] = None
    # Region-level destination (e.g. "europe", "western_europe") used by
    # the open-jaw search tool when no single destination was resolved.
    destination_region: Optional[str] = None
    outbound_date: Optional[str] = None
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    cabin_class: str = 'economy'
    max_stops: Optional[int] = None
    max_price: Optional[float] = None
    sort_by: Optional[str] = None
    # Blacklist identifiers or raw IATA codes the user asked to avoid as
    # intermediate transit hubs (e.g. ["middle_east"] or ["DXB"]).
    avoid_transit: Optional[list[str]] = None

    def as_dict(self) -> dict:
        return asdict(self)


# Slots whose dataclass default doubles as the "user didn't say" sentinel.
# Sourced from TravelIntent so the two definitions can never drift.
# max_stops is *excluded*: its default is None already, and 0 (non-stop) is a
# real user preference that must not be treated as "empty".
_UNSET_BY_DEFAULT = frozenset({'cabin_class', 'adults', 'children'})
_DEFAULTS = {
    f.name: f.default
    for f in fields(TravelIntent)
    if f.name in _UNSET_BY_DEFAULT
}


def _is_unset(key: str, value) -> bool:
    """Return True when a slot value should be treated as 'not yet provided'.

    ``max_stops`` is deliberately excluded from the defaults table so that
    ``0`` (non-stop) is treated as a real user preference, not as "empty".
    Empty lists also count as unset (``avoid_transit=[]`` is the same as
    "no preference").
    """
    if value is None or value == '':
        return True
    if isinstance(value, (list, tuple, set)) and len(value) == 0:
        return True
    if key in _DEFAULTS and value == _DEFAULTS[key]:
        return True
    return False


@dataclass
class DialogState:
    """Tracks accumulated intent across turns in a single conversation thread."""

    intent: TravelIntent = field(default_factory=TravelIntent)
    clarifications_asked: list[str] = field(default_factory=list)

    def merge(self, new: TravelIntent) -> None:
        """Fill in any fields on the tracked intent that the new turn resolved.

        A value from ``new`` replaces the tracked value only when the
        tracked slot is still unset. This preserves earlier turns: if the
        user said "2026-05-01" on turn 1, a later turn that omits a date
        will not clobber it.
        """
        for key, value in new.as_dict().items():
            if _is_unset(key, value):
                continue
            current = getattr(self.intent, key)
            if _is_unset(key, current):
                setattr(self.intent, key, value)


REQUIRED_SLOTS = ('origin_code', 'destination_code', 'outbound_date')


def missing_slots(intent: TravelIntent) -> list[str]:
    """Return the list of required slots the intent is still missing.

    "Destination" is satisfied by *either* a single city (``destination_code``,
    the normal flights_finder path) or a region (``destination_region``, the
    open-jaw path). The clarifier only asks when both are empty.
    """
    missing: list[str] = []
    if not intent.origin_code:
        missing.append('origin_code')
    if not intent.destination_code and not intent.destination_region:
        missing.append('destination_code')
    if not intent.outbound_date:
        missing.append('outbound_date')
    return missing


def clarification_question(slot: str) -> str:
    prompts = {
        'origin_code': 'Which city will you be flying from?',
        'destination_code': 'Which city or region would you like to fly to?',
        'outbound_date': 'What date do you want to depart? (e.g. 2026-05-01 or "next weekend")',
    }
    return prompts.get(slot, f'Could you clarify: {slot}?')


# ---------------------------------------------------------------------------
# Heuristic extraction
# ---------------------------------------------------------------------------

# The terminator clause lists every word that should *end* a city name
# capture. Anything placed here will not be accidentally glued onto the city,
# e.g. "to Tokyo direct on 2026-05-01" would otherwise capture "Tokyo direct".
_TERMINATORS = (
    r'on|in|next|this|for|direct|non-?stop|nonstop|cheap|cheapest|'
    r'business|economy|first|premium|with|via|by'
)
# City body: Unicode letters (covers accented Latin like São Paulo / München),
# spaces, hyphens, apostrophes, and digits (for terminal codes like "T3").
# Excludes the terminator words via the boundary group that follows.
_CITY_CHARS = r"[^\W_][\w '\-]*?"
_FROM_TO_EN = re.compile(
    rf'from\s+({_CITY_CHARS})\s+to\s+({_CITY_CHARS})'
    rf'(?:\s+(?:{_TERMINATORS})\b|[,.?!]|$)',
    re.I | re.U,
)
# Origin-only: "from Beijing next weekend", "leaving from SFO tomorrow".
_FROM_ONLY_EN = re.compile(
    rf'(?:^|\s)from\s+({_CITY_CHARS})'
    rf'(?:\s+(?:to|{_TERMINATORS})\b|[,.?!]|$)',
    re.I | re.U,
)
# Destination-only: "fly to Tokyo", "heading to LHR", "I want to go to Paris".
_TO_ONLY_EN = re.compile(
    rf'(?:^|\s)(?:fly|go|travel|head(?:ing)?)\s+to\s+({_CITY_CHARS})'
    rf'(?:\s+(?:{_TERMINATORS})\b|[,.?!]|$)',
    re.I | re.U,
)
_FROM_TO_CN = re.compile(r'从\s*([\u4e00-\u9fa5A-Za-z ]+?)\s*(?:到|飞)\s*([\u4e00-\u9fa5A-Za-z ]+?)(?:\s|，|。|$)')
# Origin-only CJK: "从香港出发", "从北京动身"
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
    # First try the full from→to pair (either language).
    for pattern in (_FROM_TO_EN, _FROM_TO_CN):
        m = pattern.search(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    # Fall back to "from X" and "to X" in isolation. Either or both may
    # match a single turn ("fly to Tokyo" / "from Beijing next weekend").
    origin: Optional[str] = None
    destination: Optional[str] = None
    m = _FROM_ONLY_EN.search(text) or _FROM_ONLY_CN.search(text)
    if m:
        origin = m.group(1).strip()
    m = _TO_ONLY_EN.search(text)
    if m:
        destination = m.group(1).strip()
    return origin, destination


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


# Region phrases: match the longest alias first so broader substrings do not shadow it.
_REGION_EN_KEYWORDS: dict[str, str] = {
    'europe': 'europe',
    'western europe': 'western_europe',
    'northern europe': 'northern_europe',
    'southern europe': 'southern_europe',
    'central europe': 'central_europe',
}


def _extract_region(text: str) -> Optional[str]:
    """Return a canonical region key if one is mentioned, else None."""
    if not text:
        return None
    for phrase in sorted(REGION_CJK_ALIASES, key=len, reverse=True):
        if phrase in text:
            return REGION_CJK_ALIASES[phrase]
    lowered = text.lower()
    # Prefer the longest phrase so "western europe" beats "europe".
    for phrase in sorted(_REGION_EN_KEYWORDS, key=len, reverse=True):
        if phrase in lowered:
            return _REGION_EN_KEYWORDS[phrase]
    return None


# Transit blacklist phrases. The heuristic is "user said avoid X and X
# resembles a known blacklist name or an IATA code". We intentionally do
# not try to catch every possible phrasing — the LLM can fall back to
# passing avoid_transit explicitly.
_AVOID_VERBS_EN = re.compile(
    r'(?:avoid|without|no|not?)\s+([a-z][a-z \-]+?)\s*(?:transit|connection|layover|stopover|\.|,|$)',
    re.I,
)
_AVOID_CN_TO_BLACKLIST: dict[str, str] = {
    '不要中东中转': 'middle_east',
    '不经中东': 'middle_east',
    '避免中东': 'middle_east',
    '不要中东': 'middle_east',
    '不要土耳其': 'middle_east_strict',
    '不经土耳其': 'middle_east_strict',
}


def _extract_avoid_transit(text: str) -> Optional[list[str]]:
    """Return a list of blacklist names / IATA codes mentioned by the user."""
    if not text:
        return None
    matches: list[str] = []

    # CJK: match explicit phrases first.
    for phrase, canonical in _AVOID_CN_TO_BLACKLIST.items():
        if phrase in text:
            matches.append(canonical)

    # English: "avoid X transit" / "no X connection" / "without X stopover".
    for m in _AVOID_VERBS_EN.finditer(text):
        candidate = m.group(1).strip().lower().replace(' ', '_').replace('-', '_')
        if candidate in TRANSIT_BLACKLISTS:
            matches.append(candidate)
            continue
        # Bare IATA: "avoid DXB" / "no DOH"
        upper = m.group(1).strip().upper()
        if len(upper) == 3 and upper.isalpha():
            matches.append(upper)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered or None


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

    # Region-level destination (e.g. "欧洲", "western europe"). When the
    # user also gave a specific city the single-city destination wins;
    # open-jaw search only kicks in when ``destination_code`` is None.
    region = _extract_region(text)
    if region:
        intent.destination_region = region

    avoid = _extract_avoid_transit(text)
    if avoid:
        intent.avoid_transit = avoid

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
