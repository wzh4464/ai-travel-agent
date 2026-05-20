"""Unit tests for fuzzy date/price/stops interpreters."""

from __future__ import annotations

import datetime

import pytest

from agents.intent.fuzzy import (
    interpret_fuzzy_date,
    interpret_price_preference,
    interpret_stops_preference,
)


# A known Friday so weekday-relative phrases are deterministic.
FRIDAY = datetime.date(2026, 4, 10)


class TestInterpretFuzzyDate:
    def test_iso_passthrough(self):
        result = interpret_fuzzy_date('2026-05-01')
        assert result is not None
        assert result.start == '2026-05-01'
        assert result.end is None

    def test_iso_range(self):
        result = interpret_fuzzy_date('2026-05-01 to 2026-05-08')
        assert result.start == '2026-05-01'
        assert result.end == '2026-05-08'

    def test_today(self):
        result = interpret_fuzzy_date('today', today=FRIDAY)
        assert result.start == '2026-04-10'

    def test_tomorrow(self):
        result = interpret_fuzzy_date('tomorrow', today=FRIDAY)
        assert result.start == '2026-04-11'

    def test_next_weekend_is_after_this_weekend(self):
        result = interpret_fuzzy_date('next weekend', today=FRIDAY)
        # next weekend from a Friday should land on the Saturday a week later
        assert result.start == '2026-04-18'
        assert result.end == '2026-04-19'

    def test_this_weekend(self):
        result = interpret_fuzzy_date('this weekend', today=FRIDAY)
        assert result.start == '2026-04-11'
        assert result.end == '2026-04-12'

    def test_next_week(self):
        result = interpret_fuzzy_date('next week', today=FRIDAY)
        # Monday of the following week
        assert result.start == '2026-04-13'

    def test_end_of_month(self):
        result = interpret_fuzzy_date('end of month', today=FRIDAY)
        assert result.start == '2026-04-30'

    def test_cjk_xia_zhou_mo(self):
        result = interpret_fuzzy_date('下周末', today=FRIDAY)
        assert result is not None

    def test_cjk_ming_tian(self):
        result = interpret_fuzzy_date('明天', today=FRIDAY)
        assert result.start == '2026-04-11'

    def test_cjk_3_days_later(self):
        result = interpret_fuzzy_date('3天后', today=FRIDAY)
        assert result.start == '2026-04-13'

    def test_in_n_days(self):
        result = interpret_fuzzy_date('in 5 days', today=FRIDAY)
        assert result.start == '2026-04-15'

    def test_unknown_phrase_returns_none(self):
        assert interpret_fuzzy_date('sometime-ish') is None

    def test_empty_returns_none(self):
        assert interpret_fuzzy_date('') is None


class TestInterpretPricePreference:
    def test_under_numeric(self):
        assert interpret_price_preference('under $500') == {'max_price': 500.0}

    def test_explicit_less_than(self):
        assert interpret_price_preference('<= 800') == {'max_price': 800.0}

    def test_cheap_phrase(self):
        assert interpret_price_preference('I want the cheapest one') == {'sort_by': 'price'}

    def test_fastest_phrase(self):
        assert interpret_price_preference('fastest please') == {'sort_by': 'duration'}

    def test_cjk_budget(self):
        assert interpret_price_preference('便宜的机票') == {'sort_by': 'price'}

    def test_cjk_under_500(self):
        assert interpret_price_preference('500美元以内') == {'max_price': 500.0}

    def test_dollar_amount_or_less(self):
        # Regression: the "$N or less" / "$N max" branch must beat the
        # generic "cheap" keyword fallback.
        assert interpret_price_preference('$500 or less') == {'max_price': 500.0}

    def test_no_signal(self):
        assert interpret_price_preference('random noise') is None


class TestInterpretStopsPreference:
    @pytest.mark.parametrize('phrase', ['non-stop', 'nonstop', 'direct flight please', '直飞', '直达航班'])
    def test_direct_maps_to_zero(self, phrase):
        assert interpret_stops_preference(phrase) == 0

    def test_one_stop_ok(self):
        assert interpret_stops_preference('at most 1 stop') == 1

    def test_no_signal(self):
        assert interpret_stops_preference('whatever') is None
