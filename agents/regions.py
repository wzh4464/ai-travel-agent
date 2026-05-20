"""Geographic region definitions and transit-hub blacklists.

Used by the open-jaw search tool to expand phrases like "Europe" or
"北欧" into a concrete list of IATA airport codes, and to translate
user preferences like "不要中东中转" into the set of airport codes to
exclude from itineraries.

All lists here are curated by hand. In a production setup they would
live behind a versioned reference-data service (OurAirports + Amadeus
reference-data) so the set of "European hubs" can drift without a code
release.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Region → IATA codes
# ---------------------------------------------------------------------------

# Western + Southern + Northern + Central Europe: the set of major
# international hubs a traveller from East Asia would realistically land
# at. Deliberately excludes IST / SAW (often grouped with Middle East),
# Russian airports, and most Eastern Europe smaller airports.
_EUROPE_CORE: list[str] = [
    # UK & Ireland
    'LHR', 'LGW', 'DUB',
    # France
    'CDG', 'ORY',
    # Spain & Portugal
    'MAD', 'BCN', 'LIS',
    # Italy
    'FCO', 'MXP',
    # DACH
    'FRA', 'MUC', 'BER', 'ZRH', 'VIE',
    # Benelux
    'AMS', 'BRU',
    # Nordics
    'CPH', 'ARN', 'OSL', 'HEL',
]

_EUROPE_EXTENDED: list[str] = _EUROPE_CORE + [
    # Less-common hubs
    'ATH', 'PRG',
    'STN', 'LCY', 'LIN', 'BGY',
]

_WESTERN_EUROPE: list[str] = [
    'LHR', 'LGW', 'DUB', 'CDG', 'ORY', 'AMS', 'BRU',
    'FRA', 'MUC', 'BER', 'ZRH', 'VIE',
]

_SOUTHERN_EUROPE: list[str] = [
    'MAD', 'BCN', 'LIS', 'FCO', 'MXP', 'ATH',
]

_NORTHERN_EUROPE: list[str] = ['CPH', 'ARN', 'OSL', 'HEL']

_CENTRAL_EUROPE: list[str] = ['FRA', 'MUC', 'BER', 'VIE', 'PRG', 'ZRH']


REGIONS: dict[str, list[str]] = {
    'europe': _EUROPE_CORE,
    'europe_extended': _EUROPE_EXTENDED,
    'western_europe': _WESTERN_EUROPE,
    'southern_europe': _SOUTHERN_EUROPE,
    'northern_europe': _NORTHERN_EUROPE,
    'central_europe': _CENTRAL_EUROPE,
}

REGION_CJK_ALIASES: dict[str, str] = {
    '欧洲': 'europe',
    '全欧洲': 'europe_extended',
    '西欧': 'western_europe',
    '南欧': 'southern_europe',
    '北欧': 'northern_europe',
    '中欧': 'central_europe',
}


def expand_region(name: str) -> list[str]:
    """Return the canonical IATA list for ``name`` (empty if unknown)."""
    name = (name or '').strip()
    if not name:
        return []
    if name in REGION_CJK_ALIASES:
        return REGIONS.get(REGION_CJK_ALIASES[name], [])
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key in REGIONS:
        return REGIONS[key]
    return []


# ---------------------------------------------------------------------------
# Transit-hub blacklists
# ---------------------------------------------------------------------------

# Default "no Middle East" — includes the big Gulf carriers' hubs. IST
# and SAW (Turkey) are excluded by default because Turkish Airlines is
# often not considered a "Middle East" transit by East Asian travellers.
_MIDDLE_EAST: list[str] = [
    'DXB', 'DWC',   # Dubai
    'AUH',          # Abu Dhabi
    'DOH',          # Doha
    'RUH', 'JED',   # Saudi Arabia
    'KWI',          # Kuwait
    'BAH',          # Bahrain
    'MCT',          # Muscat
]

# Strict variant that additionally excludes Istanbul (useful when users
# specifically do not want a Turkish Airlines connection).
_MIDDLE_EAST_STRICT: list[str] = _MIDDLE_EAST + ['IST', 'SAW']


TRANSIT_BLACKLISTS: dict[str, list[str]] = {
    'middle_east': _MIDDLE_EAST,
    'middle_east_strict': _MIDDLE_EAST_STRICT,
}

TRANSIT_CJK_ALIASES: dict[str, str] = {
    '中东': 'middle_east',
    '中东严格': 'middle_east_strict',
    '中东含土耳其': 'middle_east_strict',
}


def expand_transit_blacklist(name: str) -> set[str]:
    """Translate a blacklist name ("middle_east", "中东") into IATA codes.

    Unknown names fall back to ``{name.upper()}`` so callers can pass a
    raw airport code like ``DXB`` directly.
    """
    if not name:
        return set()
    if name in TRANSIT_CJK_ALIASES:
        return set(TRANSIT_BLACKLISTS[TRANSIT_CJK_ALIASES[name]])
    key = name.strip().lower().replace(' ', '_').replace('-', '_')
    if key in TRANSIT_BLACKLISTS:
        return set(TRANSIT_BLACKLISTS[key])
    return {name.strip().upper()}
