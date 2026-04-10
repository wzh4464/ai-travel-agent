"""Base classes and cross-cutting concerns for flight data sources.

Handles authentication (delegated to concrete subclasses), rate limiting via a
simple token bucket, and exponential backoff retries on transient failures.
"""

from __future__ import annotations

import abc
import functools
import threading
import time
from typing import Any, Callable, TypeVar

from agents.errors import RateLimitedError, UpstreamAPIError

T = TypeVar('T')


class RateLimiter:
    """Thread-safe token bucket rate limiter.

    The agent may issue several concurrent tool calls in a single LLM turn, so
    we serialise access across threads to stay within the provider's quota.
    """

    def __init__(self, rate_per_second: float, burst: int | None = None):
        self.rate = float(rate_per_second)
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_second)))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = (tokens - self._tokens) / self.rate
            time.sleep(needed)


def with_retries(
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (UpstreamAPIError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential-backoff retry decorator.

    Only retries on ``retry_on`` exceptions. :class:`RateLimitedError`
    instances are honoured via the ``retry_after`` hint when present.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    wait = delay
                    if isinstance(exc, RateLimitedError) and exc.retry_after:
                        wait = max(wait, exc.retry_after)
                    time.sleep(min(wait, max_delay))
                    delay = min(delay * 2, max_delay)

        return wrapper

    return decorator


class BaseFlightSource(abc.ABC):
    """Abstract flight-data provider interface.

    Concrete implementations (SerpAPI, Amadeus, Kiwi, ...) translate the
    canonical search parameters into provider-specific API calls and return
    results already normalised to :class:`agents.data_sources.normalizer.Flight`.
    """

    name: str = 'base'

    def __init__(self, rate_limiter: RateLimiter | None = None):
        self.rate_limiter = rate_limiter or RateLimiter(rate_per_second=2.0, burst=4)

    def is_configured(self) -> bool:
        """Return True when the adapter has the credentials it needs to run.

        Subclasses override this so the aggregator can skip unconfigured
        sources silently instead of surfacing auth errors to the user.
        """
        return True

    @abc.abstractmethod
    def search(
        self,
        *,
        origin: str,
        destination: str,
        outbound_date: str,
        return_date: str | None = None,
        adults: int = 1,
        children: int = 0,
        infants_in_seat: int = 0,
        infants_on_lap: int = 0,
        cabin_class: str = 'economy',
        max_stops: int | None = None,
    ) -> list[dict]:
        """Run a flight search and return normalised Flight dicts."""

    @abc.abstractmethod
    def details(self, flight_id: str) -> dict:
        """Fetch baggage / fare / change-policy information for a flight."""
