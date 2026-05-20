"""Unit tests for the error hierarchy and degrade() payload shape."""

from __future__ import annotations

from agents.errors import (
    AmbiguousInputError,
    InvalidParameterError,
    MissingParameterError,
    NoResultsError,
    RateLimitedError,
    TravelAgentError,
    UpstreamAPIError,
    degrade,
)


class TestErrorHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(MissingParameterError, TravelAgentError)
        assert issubclass(InvalidParameterError, TravelAgentError)
        assert issubclass(AmbiguousInputError, TravelAgentError)
        assert issubclass(UpstreamAPIError, TravelAgentError)
        assert issubclass(RateLimitedError, UpstreamAPIError)
        assert issubclass(NoResultsError, TravelAgentError)

    def test_rate_limited_sets_status_429(self):
        exc = RateLimitedError('amadeus', retry_after=5.0)
        assert exc.status == 429
        assert exc.retry_after == 5.0

    def test_no_results_has_helpful_user_message(self):
        exc = NoResultsError('JFK', 'LHR', '2026-05-01')
        assert 'JFK' in exc.user_message
        assert 'LHR' in exc.user_message


class TestDegrade:
    def test_missing_parameter_shape(self):
        payload = degrade(MissingParameterError(['outbound_date']))
        assert payload['status'] == 'error'
        assert payload['error_type'] == 'MissingParameterError'
        assert 'outbound_date' in payload['user_message']

    def test_invalid_parameter_shape(self):
        payload = degrade(InvalidParameterError('destination_region', 'atlantis'))
        assert payload['status'] == 'error'
        assert payload['error_type'] == 'InvalidParameterError'
        assert 'destination_region' in payload['user_message']

    def test_upstream_error_scrubs_free_form_details(self):
        # Embed an email in the upstream error — should not reach the LLM.
        exc = UpstreamAPIError('amadeus', status=500, detail='contact ops@example.com')
        payload = degrade(exc)
        assert payload['error_type'] == 'UpstreamAPIError'
        assert 'ops@example.com' not in payload['details']
        assert '[REDACTED]' in payload['details']

    def test_rate_limited_degrades(self):
        payload = degrade(RateLimitedError('kiwi', retry_after=2.0))
        assert payload['error_type'] == 'RateLimitedError'
        assert 'kiwi' in payload['details'].lower() or 'rate limited' in payload['details'].lower()

    def test_unknown_exception_becomes_unknown_error(self):
        payload = degrade(RuntimeError('something weird'))
        assert payload['error_type'] == 'UnknownError'
        assert payload['user_message']

    def test_user_message_of_missing_is_scrubbed(self):
        # A future caller might pass a PII-flavoured field name — verify scrub
        # still runs on user_message.
        exc = MissingParameterError(['contact email (user@test.com)'])
        payload = degrade(exc)
        assert 'user@test.com' not in payload['user_message']
