"""Unit tests for the token-bucket rate limiter."""

from __future__ import annotations

import threading
import time

from agents.data_sources.base import RateLimiter


class TestRateLimiter:
    def test_burst_is_free(self, monkeypatch):
        """Acquiring up to the burst size must never sleep — checked
        by asserting :func:`time.sleep` is not invoked, so the test is
        deterministic on slow / contended CI runners."""
        rl = RateLimiter(rate_per_second=100, burst=5)
        sleep_calls: list[float] = []
        monkeypatch.setattr(
            'agents.data_sources.base.time.sleep',
            lambda secs: sleep_calls.append(secs),
        )
        for _ in range(5):
            rl.acquire()
        assert sleep_calls == []

    def test_blocks_once_burst_exhausted(self):
        rl = RateLimiter(rate_per_second=10, burst=2)
        # Drain the bucket
        rl.acquire()
        rl.acquire()
        start = time.monotonic()
        rl.acquire()  # third call has to wait ~0.1s for a token
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # allow scheduling slack

    def test_thread_safe(self):
        rl = RateLimiter(rate_per_second=1000, burst=10)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(5):
                    rl.acquire()
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
