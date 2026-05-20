"""Unit tests for the IATA lookup helper."""

from __future__ import annotations

from agents.intent.iata import CITY_TO_IATA, lookup


class TestLookup:
    def test_exact_city(self):
        assert lookup('new york') == CITY_TO_IATA['new york']
        assert 'JFK' in lookup('new york')

    def test_case_insensitive(self):
        assert lookup('NEW YORK') == lookup('new york')

    def test_cjk_alias(self):
        assert 'PEK' in lookup('北京')
        assert 'HND' in lookup('东京') or 'NRT' in lookup('东京')

    def test_3_letter_passthrough(self):
        assert lookup('jfk') == ['JFK']
        assert lookup('LHR') == ['LHR']

    def test_substring_fallback(self):
        # "London Heathrow" -> contains 'london' substring
        assert 'LHR' in lookup('london heathrow')

    def test_unknown_city_returns_empty(self):
        assert lookup('nowhere-ville') == []

    def test_empty_input(self):
        assert lookup('') == []
        assert lookup(None) == []
