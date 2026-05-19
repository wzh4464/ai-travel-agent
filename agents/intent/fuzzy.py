"""Interpret fuzzy natural-language inputs into concrete parameters.

These helpers cover the three most common sources of ambiguity reported by
users:

* Fuzzy dates: "next weekend", "下周末", "tomorrow", "月底".
* Fuzzy price preferences: "cheap", "便宜的", "under $500".
* Fuzzy routing preferences: "direct", "non-stop", "直飞", "transfer ok".
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DateRange:
    start: str  # YYYY-MM-DD
    end: Optional[str] = None  # YYYY-MM-DD, inclusive


_WEEKDAY_EN = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6,
}
_WEEKDAY_CN = {
    '周一': 0, '周二': 1, '周三': 2, '周四': 3, '周五': 4, '周六': 5, '周日': 6,
    '星期一': 0, '星期二': 1, '星期三': 2, '星期四': 3, '星期五': 4, '星期六': 5, '星期日': 6,
}


def _fmt(d: datetime.date) -> str:
    return d.isoformat()


def interpret_fuzzy_date(
    text: str,
    *,
    today: Optional[datetime.date] = None,
) -> Optional[DateRange]:
    """Resolve common fuzzy date expressions to a concrete range.

    Returns ``None`` when the phrase is not recognised, so callers can fall
    back to prompting the user for clarification.
    """
    if not text:
        return None
    today = today or datetime.date.today()
    s = text.strip().lower()

    # ISO date passthrough: "2026-05-01" or "2026-05-01 to 2026-05-08"
    iso = re.findall(r'\d{4}-\d{2}-\d{2}', s)
    if iso:
        return DateRange(start=iso[0], end=iso[1] if len(iso) > 1 else None)

    # Compact Chinese-style short range: "4.23-5.3", "12/25-1/2", "4.23到5.3"
    # Year defaults to the current year, rolling forward to next year if
    # the start date has already passed.
    compact = re.search(
        r'(\d{1,2})[./](\d{1,2})\s*[-~到至]\s*(\d{1,2})[./](\d{1,2})',
        s,
    )
    if compact:
        try:
            mo1, d1, mo2, d2 = (int(g) for g in compact.groups())
            year = today.year
            start = datetime.date(year, mo1, d1)
            if start < today:
                start = datetime.date(year + 1, mo1, d1)
            end = datetime.date(start.year, mo2, d2)
            if end < start:
                end = datetime.date(start.year + 1, mo2, d2)
            return DateRange(start=_fmt(start), end=_fmt(end))
        except ValueError:
            pass

    if s in ('today', '今天'):
        return DateRange(start=_fmt(today))
    if s in ('tomorrow', '明天'):
        return DateRange(start=_fmt(today + datetime.timedelta(days=1)))
    if s in ('day after tomorrow', '后天'):
        return DateRange(start=_fmt(today + datetime.timedelta(days=2)))

    if 'next weekend' in s or '下周末' in s or '下个周末' in s:
        days_to_sat = (5 - today.weekday()) % 7 or 7
        sat = today + datetime.timedelta(days=days_to_sat + 7)
        return DateRange(start=_fmt(sat), end=_fmt(sat + datetime.timedelta(days=1)))

    if 'this weekend' in s or '这周末' in s or '本周末' in s:
        days_to_sat = (5 - today.weekday()) % 7
        sat = today + datetime.timedelta(days=days_to_sat)
        return DateRange(start=_fmt(sat), end=_fmt(sat + datetime.timedelta(days=1)))

    if 'next week' in s or '下周' in s:
        monday = today + datetime.timedelta(days=(7 - today.weekday()))
        return DateRange(start=_fmt(monday), end=_fmt(monday + datetime.timedelta(days=6)))

    if 'end of month' in s or '月底' in s:
        # last day of current month
        first_next = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        last = first_next - datetime.timedelta(days=1)
        return DateRange(start=_fmt(last))

    # "next monday" / "下周一"
    for table in (_WEEKDAY_EN, _WEEKDAY_CN):
        for key, idx in table.items():
            if key in s:
                offset = (idx - today.weekday()) % 7
                offset = offset or 7  # always forward
                if 'next' in s or '下' in s:
                    offset += 7
                return DateRange(start=_fmt(today + datetime.timedelta(days=offset)))

    # "in 3 days" / "3天后"
    m = re.search(r'in\s+(\d+)\s+day', s) or re.search(r'(\d+)\s*天后', s)
    if m:
        return DateRange(start=_fmt(today + datetime.timedelta(days=int(m.group(1)))))

    return None


def interpret_price_preference(text: str) -> Optional[dict]:
    """Map price-preference phrases to concrete filters."""
    if not text:
        return None
    s = text.lower()

    # Price-first phrases: "500美元以内", "800元以下", "$500 or less"
    m = re.search(r'\$?(\d+)\s*(?:美元|元|块|rmb|usd)?\s*(?:以内|以下|左右|or less)', s)
    if m:
        return {'max_price': float(m.group(1))}
    # Keyword-first phrases: "under $500", "below 800", "<= 800"
    m = re.search(r'(?:under|below|<=?)\s*\$?(\d+)', s)
    if m:
        return {'max_price': float(m.group(1))}
    # "$500 ... cheap" — dollar amount with a cheap/less hint nearby
    m = re.search(r'\$(\d+)', s)
    if m and ('cheap' in s or 'less' in s):
        return {'max_price': float(m.group(1))}

    if any(k in s for k in ('cheap', 'cheapest', 'low cost', 'budget', '便宜', '最便宜', '经济')):
        return {'sort_by': 'price'}
    if any(k in s for k in ('fastest', 'quickest', '最快', '时间最短')):
        return {'sort_by': 'duration'}
    return None


def interpret_stops_preference(text: str) -> Optional[int]:
    """Return a ``max_stops`` integer from phrases like "direct" or "直飞"."""
    if not text:
        return None
    s = text.lower()
    if any(k in s for k in ('non-stop', 'nonstop', 'direct', '直飞', '直达')):
        return 0
    if 'at most 1 stop' in s or '最多一次中转' in s or 'one stop' in s:
        return 1
    return None
