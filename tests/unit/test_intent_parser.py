"""Unit tests for the deterministic intent extractor and dialog state."""

from __future__ import annotations

import datetime

from agents.intent.parser import (
    DialogState,
    TravelIntent,
    clarification_question,
    extract_intent,
    missing_slots,
)


class TestExtractIntentEnglish:
    def test_basic_from_to(self):
        intent = extract_intent('I want to fly from Beijing to New York next weekend')
        assert intent.origin_code == 'PEK'
        assert intent.destination_code == 'JFK'
        assert intent.outbound_date is not None
        assert intent.origin_city == 'Beijing'
        assert intent.destination_city == 'New York'

    def test_preferences_direct_and_cheap(self):
        intent = extract_intent(
            'I want to fly from Beijing to New York next weekend, non-stop, cheap'
        )
        assert intent.max_stops == 0
        assert intent.sort_by == 'price'

    def test_cabin_class_business(self):
        intent = extract_intent('Fly from Madrid to Tokyo on 2026-05-01, business class')
        assert intent.cabin_class == 'business'
        assert intent.origin_code == 'MAD'
        assert intent.destination_code in ('HND', 'NRT')
        assert intent.outbound_date == '2026-05-01'

    def test_passenger_count(self):
        intent = extract_intent('from Paris to Rome next week for 3 adults')
        assert intent.adults == 3

    def test_under_price_ceiling(self):
        intent = extract_intent('cheap flight from London to Berlin under $300')
        assert intent.max_price == 300.0

    def test_returns_empty_on_empty_input(self):
        intent = extract_intent('')
        assert intent.origin_code is None
        assert intent.destination_code is None


class TestExtractIntentChinese:
    def test_cjk_from_to(self):
        intent = extract_intent('从北京到东京，下周一，2人，商务舱')
        assert intent.origin_code == 'PEK'
        assert intent.destination_code in ('HND', 'NRT')
        assert intent.cabin_class == 'business'
        assert intent.adults == 2

    def test_cjk_direct_flight(self):
        intent = extract_intent('从上海飞首尔，直飞，2026-06-01')
        assert intent.origin_code in ('PVG', 'SHA')
        assert intent.destination_code in ('ICN', 'GMP')
        assert intent.max_stops == 0
        assert intent.outbound_date == '2026-06-01'


class TestExtractIntentRegionAndTransit:
    """Tests that demonstrate the open-jaw query path.

    The canonical user phrase for this feature is
    ``"从香港出发去欧洲 4.23-5.3 要便宜 不要中东中转"`` — everything in
    that one sentence should land as structured slots.
    """

    def test_full_hong_kong_europe_sentence(self):
        intent = extract_intent('从香港出发去欧洲 4.23-5.3 要便宜 不要中东中转')
        assert intent.origin_code == 'HKG'
        # No specific destination city (region query)
        assert intent.destination_code is None
        assert intent.destination_region == 'europe'
        assert intent.outbound_date == '2026-04-23'
        assert intent.return_date == '2026-05-03'
        assert intent.sort_by == 'price'
        assert intent.avoid_transit == ['middle_east']

    def test_english_region_phrase(self):
        intent = extract_intent(
            'I want to go somewhere in western europe from 2026-04-23 to 2026-05-03'
        )
        assert intent.destination_region == 'western_europe'

    def test_region_longer_phrase_wins(self):
        intent = extract_intent('visiting northern europe next month')
        assert intent.destination_region == 'northern_europe'

    def test_no_region_when_not_mentioned(self):
        intent = extract_intent('from Beijing to Tokyo on 2026-05-01')
        assert intent.destination_region is None

    def test_avoid_transit_raw_iata(self):
        intent = extract_intent('fly to London but avoid DXB transit please')
        assert 'DXB' in (intent.avoid_transit or [])

    def test_avoid_transit_strict_variant(self):
        intent = extract_intent('我不要土耳其中转')
        assert intent.avoid_transit == ['middle_east_strict']

    def test_origin_only_cjk(self):
        intent = extract_intent('从香港出发明天')
        assert intent.origin_code == 'HKG'

    def test_chinese_short_date_range(self):
        import datetime
        from agents.intent.fuzzy import interpret_fuzzy_date
        # Before 4/23 in the current year, we want the current year dates.
        result = interpret_fuzzy_date('4.23-5.3', today=datetime.date(2026, 4, 1))
        assert result.start == '2026-04-23'
        assert result.end == '2026-05-03'

    def test_chinese_short_date_range_rolls_forward(self):
        import datetime
        from agents.intent.fuzzy import interpret_fuzzy_date
        # After 4/23 this year, should roll to next year.
        result = interpret_fuzzy_date('4.23-5.3', today=datetime.date(2026, 6, 1))
        assert result.start == '2027-04-23'
        assert result.end == '2027-05-03'

    def test_chinese_short_date_with_cjk_separator(self):
        import datetime
        from agents.intent.fuzzy import interpret_fuzzy_date
        result = interpret_fuzzy_date('4.23到5.3', today=datetime.date(2026, 4, 1))
        assert result.start == '2026-04-23'
        assert result.end == '2026-05-03'

    def test_avoid_transit_empty_list_is_none(self):
        intent = extract_intent('just a normal search')
        assert intent.avoid_transit is None


class TestDialogState:
    def test_merge_fills_in_missing_slots(self):
        state = DialogState()
        state.merge(extract_intent('fly to Tokyo'))
        assert state.intent.destination_code in ('HND', 'NRT')
        state.merge(extract_intent('from Beijing next weekend'))
        assert state.intent.origin_code == 'PEK'
        assert state.intent.outbound_date is not None

    def test_merge_does_not_clobber_existing_values(self):
        state = DialogState()
        state.merge(extract_intent('from Beijing to Tokyo on 2026-05-01'))
        first_date = state.intent.outbound_date
        state.merge(extract_intent('business class please'))
        # Dates should not be overwritten by an unrelated turn.
        assert state.intent.outbound_date == first_date
        assert state.intent.cabin_class == 'business'

    def test_max_stops_zero_propagates(self):
        state = DialogState()
        state.merge(extract_intent('from Beijing to Tokyo direct on 2026-05-01'))
        assert state.intent.max_stops == 0


class TestMissingSlots:
    def test_all_present(self):
        intent = TravelIntent(
            origin_code='PEK', destination_code='JFK', outbound_date='2026-05-01'
        )
        assert missing_slots(intent) == []

    def test_missing_origin(self):
        intent = TravelIntent(destination_code='JFK', outbound_date='2026-05-01')
        assert missing_slots(intent) == ['origin_code']

    def test_missing_multiple(self):
        assert missing_slots(TravelIntent()) == [
            'origin_code', 'destination_code', 'outbound_date'
        ]

    def test_clarification_questions_exist_for_all_slots(self):
        for slot in ('origin_code', 'destination_code', 'outbound_date'):
            assert clarification_question(slot)
            assert '?' in clarification_question(slot)
