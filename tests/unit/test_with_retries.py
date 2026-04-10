"""Unit tests for the exponential-backoff retry decorator."""

from __future__ import annotations

import time

import pytest

from agents.data_sources.base import with_retries
from agents.errors import RateLimitedError, UpstreamAPIError


class TestWithRetries:
    def test_returns_value_on_success(self):
        @with_retries(max_attempts=3, base_delay=0.0, max_delay=0.0)
        def ok():
            return 42

        assert ok() == 42

    def test_retries_on_upstream_error_then_succeeds(self):
        attempts = {'n': 0}

        @with_retries(max_attempts=4, base_delay=0.0, max_delay=0.0)
        def flaky():
            attempts['n'] += 1
            if attempts['n'] < 3:
                raise UpstreamAPIError('test', status=500)
            return 'ok'

        assert flaky() == 'ok'
        assert attempts['n'] == 3

    def test_raises_after_max_attempts(self):
        attempts = {'n': 0}

        @with_retries(max_attempts=3, base_delay=0.0, max_delay=0.0)
        def always_fails():
            attempts['n'] += 1
            raise UpstreamAPIError('test', status=500)

        with pytest.raises(UpstreamAPIError):
            always_fails()
        assert attempts['n'] == 3

    def test_does_not_retry_on_unlisted_exceptions(self):
        attempts = {'n': 0}

        @with_retries(max_attempts=3, base_delay=0.0, max_delay=0.0)
        def value_error():
            attempts['n'] += 1
            raise ValueError('boom')

        with pytest.raises(ValueError):
            value_error()
        assert attempts['n'] == 1

    def test_honours_rate_limited_retry_after(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr('agents.data_sources.base.time.sleep', sleeps.append)

        attempts = {'n': 0}

        @with_retries(max_attempts=3, base_delay=0.1, max_delay=10.0)
        def limited():
            attempts['n'] += 1
            if attempts['n'] == 1:
                raise RateLimitedError('test', retry_after=2.5)
            return 'ok'

        assert limited() == 'ok'
        # First retry should have waited ~2.5s (the Retry-After hint)
        assert sleeps
        assert sleeps[0] >= 2.5
