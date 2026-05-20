"""Custom exceptions and graceful-degradation helpers for the travel agent."""

from __future__ import annotations

from agents.privacy import scrub


class TravelAgentError(Exception):
    """Base class for all recoverable errors raised by the agent."""

    user_message: str = 'Something went wrong. Please try again.'

    def __init__(self, message: str | None = None, *, user_message: str | None = None):
        super().__init__(message or self.user_message)
        if user_message is not None:
            self.user_message = user_message


class MissingParameterError(TravelAgentError):
    """Raised when required structured parameters are missing from the request."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        msg = f"Missing required parameters: {', '.join(missing)}"
        super().__init__(msg, user_message=msg)


class InvalidParameterError(TravelAgentError):
    """Raised when a structured parameter is present but invalid."""

    def __init__(self, field: str, value: str, reason: str | None = None):
        self.field = field
        self.value = value
        suffix = f': {reason}' if reason else ''
        msg = f"Invalid value for '{field}': {value}{suffix}"
        super().__init__(msg, user_message=msg)


class AmbiguousInputError(TravelAgentError):
    """Raised when user intent cannot be resolved without clarification."""

    def __init__(self, field: str, options: list[str] | None = None):
        self.field = field
        self.options = options or []
        suffix = f" Options: {', '.join(self.options)}" if self.options else ''
        super().__init__(f"Ambiguous value for '{field}'.{suffix}")


class UpstreamAPIError(TravelAgentError):
    """Raised when an upstream flight data provider fails."""

    def __init__(self, provider: str, status: int | None = None, detail: str = ''):
        self.provider = provider
        self.status = status
        super().__init__(
            f'{provider} failed (status={status}): {detail}',
            user_message=f'The {provider} flight search is temporarily unavailable.',
        )


class RateLimitedError(UpstreamAPIError):
    """Raised when the upstream API signals throttling."""

    def __init__(self, provider: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(provider, status=429, detail='rate limited')


class NoResultsError(TravelAgentError):
    """Raised when a search returns zero flights."""

    def __init__(self, origin: str, destination: str, date: str):
        super().__init__(
            f'No flights for {origin}->{destination} on {date}',
            user_message=(
                f'No flights were found from {origin} to {destination} on {date}. '
                'Try widening your date range or allowing connections.'
            ),
        )


def degrade(exc: Exception) -> dict:
    """Convert any exception into a structured payload the LLM can reason about.

    This is the single place where upstream failures get translated into a
    machine-readable shape. The LLM can then decide whether to retry, ask the
    user a clarifying question, or surface the error message verbatim.

    All free-form fields are passed through :func:`agents.privacy.scrub` so
    PII that leaked into an upstream exception message never reaches the
    LLM context window or the tool-call log.
    """
    if isinstance(exc, TravelAgentError):
        return {
            'status': 'error',
            'error_type': exc.__class__.__name__,
            'user_message': scrub(exc.user_message),
            'details': scrub(str(exc)),
        }
    return {
        'status': 'error',
        'error_type': 'UnknownError',
        'user_message': 'An unexpected error occurred while searching flights.',
        'details': scrub(str(exc)),
    }
