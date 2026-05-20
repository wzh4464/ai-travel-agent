"""Standalone IATA lookup table and helper.

This module is intentionally dependency-free so it can be reused by the
intent parser without dragging in langchain / pydantic.
"""

from __future__ import annotations

# pylint: disable=line-too-long
CITY_TO_IATA: dict[str, list[str]] = {
    'new york': ['JFK', 'LGA', 'EWR'],
    'nyc': ['JFK', 'LGA', 'EWR'],
    'london': ['LHR', 'LGW', 'STN', 'LCY'],
    'paris': ['CDG', 'ORY'],
    'tokyo': ['HND', 'NRT'],
    'beijing': ['PEK', 'PKX'],
    'shanghai': ['PVG', 'SHA'],
    'hong kong': ['HKG'],
    'taipei': ['TPE', 'TSA'],
    'seoul': ['ICN', 'GMP'],
    'singapore': ['SIN'],
    'bangkok': ['BKK', 'DMK'],
    'dubai': ['DXB', 'DWC'],
    'istanbul': ['IST', 'SAW'],
    'madrid': ['MAD'],
    'barcelona': ['BCN'],
    'rome': ['FCO', 'CIA'],
    'milan': ['MXP', 'LIN', 'BGY'],
    'frankfurt': ['FRA'],
    'munich': ['MUC'],
    'berlin': ['BER'],
    'amsterdam': ['AMS'],
    'zurich': ['ZRH'],
    'vienna': ['VIE'],
    'moscow': ['SVO', 'DME', 'VKO'],
    'los angeles': ['LAX'],
    'san francisco': ['SFO'],
    'seattle': ['SEA'],
    'chicago': ['ORD', 'MDW'],
    'washington': ['IAD', 'DCA'],
    'boston': ['BOS'],
    'miami': ['MIA'],
    'toronto': ['YYZ', 'YTZ'],
    'vancouver': ['YVR'],
    'mexico city': ['MEX'],
    'sao paulo': ['GRU', 'CGH'],
    'buenos aires': ['EZE', 'AEP'],
    'sydney': ['SYD'],
    'melbourne': ['MEL'],
    'auckland': ['AKL'],
    'delhi': ['DEL'],
    'mumbai': ['BOM'],
    'bangalore': ['BLR'],
    'kuala lumpur': ['KUL'],
    'jakarta': ['CGK'],
    'manila': ['MNL'],
    'ho chi minh city': ['SGN'],
    'hanoi': ['HAN'],
    'doha': ['DOH'],
    'abu dhabi': ['AUH'],
    'riyadh': ['RUH'],
    'cairo': ['CAI'],
    'johannesburg': ['JNB'],
    'cape town': ['CPT'],
    'lisbon': ['LIS'],
    'copenhagen': ['CPH'],
    'stockholm': ['ARN'],
    'oslo': ['OSL'],
    'helsinki': ['HEL'],
    'dublin': ['DUB'],
    'athens': ['ATH'],
    'prague': ['PRG'],
}

CJK_ALIASES: dict[str, str] = {
    '北京': 'beijing',
    '上海': 'shanghai',
    '香港': 'hong kong',
    '台北': 'taipei',
    '东京': 'tokyo',
    '首尔': 'seoul',
    '纽约': 'new york',
    '伦敦': 'london',
    '巴黎': 'paris',
    '新加坡': 'singapore',
    '曼谷': 'bangkok',
    '迪拜': 'dubai',
}


def lookup(city_name: str) -> list[str]:
    """Return IATA codes for ``city_name`` (empty list when unknown)."""
    key = (city_name or '').strip().lower()
    if key in CJK_ALIASES:
        key = CJK_ALIASES[key]
    if key in CITY_TO_IATA:
        return CITY_TO_IATA[key]
    if len(key) == 3 and key.isalpha():
        return [key.upper()]
    for name, codes in CITY_TO_IATA.items():
        if key and key in name:
            return codes
    return []
