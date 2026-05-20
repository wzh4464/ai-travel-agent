"""Unit tests for the PII scrubbing primitives."""

from __future__ import annotations

import pytest

from agents.privacy import REDACTED, contains_pii, scrub, scrub_mapping


class TestScrubEmails:
    @pytest.mark.parametrize('sample', [
        'contact alice@example.com please',
        'Alice <alice.bob+filter@sub.example.co.uk>',
        'user_42@mail.test',
    ])
    def test_email_is_redacted(self, sample):
        out = scrub(sample)
        assert REDACTED in out
        assert '@' not in out or out.count('@') < sample.count('@')

    def test_non_email_is_unchanged(self):
        assert scrub('not an email: at sign without domain') == 'not an email: at sign without domain'


class TestScrubBearerAndApiKeys:
    def test_bearer_token_removed(self):
        out = scrub('Authorization: Bearer abc123def456ghi789')
        assert 'abc123def456ghi789' not in out
        assert REDACTED in out

    def test_api_key_assignment_removed(self):
        samples = [
            'api_key=SECRET-VALUE-abcdef1234',
            'access_token: "ASDF-1234-5678-abcd"',
            "client_secret='longsecretvalue12345'",
        ]
        for s in samples:
            out = scrub(s)
            assert REDACTED in out


class TestScrubCreditCards:
    def test_16_digit_pan_redacted(self):
        out = scrub('Charged card 4111 1111 1111 1111 today')
        assert '4111 1111 1111 1111' not in out
        assert '4111-1111-1111-1111' not in out

    def test_unbroken_pan_redacted(self):
        out = scrub('PAN=4111111111111111 ok')
        assert '4111111111111111' not in out


class TestScrubPassportAndSSN:
    def test_passport_like(self):
        assert REDACTED in scrub('passport E12345678')
        assert REDACTED in scrub('Pass: G123456789')

    def test_ssn(self):
        out = scrub('SSN 123-45-6789')
        assert '123-45-6789' not in out


class TestScrubPhone:
    def test_international_number(self):
        out = scrub('call me at +1 (415) 555-0199 anytime')
        assert '+1' not in out or '5550199' not in out.replace(' ', '')


class TestScrubNonString:
    def test_none(self):
        assert scrub(None) == ''

    def test_integer_is_stringified(self):
        assert scrub(42) == '42'

    def test_exception(self):
        exc = RuntimeError('leaked ops@example.com')
        assert 'ops@example.com' not in scrub(exc)


class TestContainsPII:
    def test_positive_cases(self):
        assert contains_pii('alice@example.com')
        assert contains_pii('Bearer abcdefg12345')
        assert contains_pii('4111 1111 1111 1111')

    def test_negative_case(self):
        assert not contains_pii('from JFK to LHR on 2026-05-01 please')


class TestScrubMapping:
    def test_redacts_string_values(self):
        out = scrub_mapping({'note': 'email me at x@y.com', 'city': 'Paris'})
        assert out['city'] == 'Paris'
        assert 'x@y.com' not in out['note']

    def test_redacts_sensitive_keys_outright(self):
        out = scrub_mapping({'passport_number': 'X12345', 'name': 'Alice'})
        assert out['passport_number'] == REDACTED
        assert out['name'] == 'Alice'

    def test_nested_dict(self):
        out = scrub_mapping({'passenger': {'email': 'a@b.com', 'name': 'Bob'}})
        assert out['passenger']['email'] == REDACTED

    def test_list_of_strings(self):
        out = scrub_mapping({'contacts': ['a@b.com', 'Bob']})
        assert REDACTED in out['contacts'][0]
        assert out['contacts'][1] == 'Bob'
