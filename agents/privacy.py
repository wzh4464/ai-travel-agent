"""PII redaction utilities.

The agent shuttles arbitrary strings back and forth between the user, tool
call arguments, upstream API errors, and the LLM. Any of those hops can leak
personally identifiable information (PII) into logs, traces, or the
conversation context that gets replayed on future turns. This module
provides a single ``scrub()`` function with a small, curated pattern
library so every layer can redact using the same rules.

Design principles
-----------------

* Regex-only. No ML/NER dependency.
* Conservative. False positives (over-redaction) are strongly preferred to
  false negatives (leakage). The tests live in ``tests/privacy``.
* Pure strings in, pure strings out — no mutation of mappings, exceptions,
  or objects. Callers decide what to wrap.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Pattern

REDACTED = '[REDACTED]'

# Tokens that, when found as a component of a key name (after lowercase +
# camelCase/snake_case split), mark the value as a credential and force
# redaction regardless of the surrounding type. Centralised here so the
# exact-match path and the tokenised path stay in sync.
_SENSITIVE_TOKENS = frozenset({
    'password', 'passport', 'token', 'secret', 'authorization',
    'email', 'phone', 'ssn', 'pan', 'cvv', 'card', 'apikey',
})


def _split_key(key: str) -> list[str]:
    """Split a key name into lowercased word tokens.

    Handles both ``snake_case`` / ``kebab-case`` (split on ``_``/``-``) and
    ``camelCase`` / ``PascalCase`` (insert a boundary before each uppercase
    letter that follows a lowercase one). Used so keys like
    ``passportNumber`` or ``authorization_header`` still match the
    blocklist.
    """
    # Insert a space before each uppercase-after-lowercase transition.
    spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', key)
    parts = re.split(r'[_\-\s]+', spaced)
    return [p.lower() for p in parts if p]


# Ordered list: more specific patterns first so they win over generic ones.
_PII_PATTERNS: list[tuple[str, Pattern[str]]] = [
    # Email addresses.
    ('EMAIL', re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
    # Bearer tokens / API keys in Authorization-ish strings.
    ('BEARER', re.compile(r'[Bb]earer\s+[A-Za-z0-9._\-+/=]{8,}')),
    ('APIKEY', re.compile(r'(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|apikey)[\"\'=: ]+[A-Za-z0-9._\-+/=]{8,}')),
    # 13-19 digit PANs (credit cards); allow spaces / dashes every 4 digits.
    ('CARD', re.compile(r'\b(?:\d[ -]?){13,19}\b')),
    # Phone numbers — deliberately conservative to avoid matching ISO dates
    # like 2026-05-01. Requires either a leading ``+`` country code or a
    # parenthesised area code to count as a phone number.
    ('PHONE', re.compile(
        r'(?:(?<!\d)\+\d[\d\-\s()]{7,}\d(?!\d))'
        r'|(?:\(\d{3}\)\s*\d{3}[-\s]?\d{4})'
    )),
    # Passport-like alphanumerics (1 letter + 7-9 digits, common formats).
    ('PASSPORT', re.compile(r'\b[A-Z][0-9]{7,9}\b')),
    # US-style SSN.
    ('SSN', re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
]


def scrub(text: Any) -> str:
    """Return ``text`` with PII replaced by ``[REDACTED]``.

    Non-string inputs are first coerced via ``str()``.
    """
    if text is None:
        return ''
    s = text if isinstance(text, str) else str(text)
    for _label, pattern in _PII_PATTERNS:
        s = pattern.sub(REDACTED, s)
    return s


def _scrub_value(value: Any, sensitive: Iterable[str]) -> Any:
    """Recursive switch used by :func:`scrub_mapping`.

    Centralising the per-type dispatch here means nested ``list`` →
    ``list`` → ``dict`` payloads (and any combination thereof) all reach
    the same scrubber, instead of the inner ``dict`` being left untouched
    by a one-level-deep comprehension.
    """
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return scrub_mapping(value, sensitive_keys=sensitive)
    if type(value) in (list, tuple):
        # Only reconstruct plain list/tuple — tuple subclasses (NamedTuple,
        # typed containers) have custom __init__ signatures that don't
        # accept a generator argument, and would crash here.
        scrubbed = [_scrub_value(v, sensitive) for v in value]
        return scrubbed if isinstance(value, list) else tuple(scrubbed)
    if isinstance(value, (list, tuple)):
        # Tuple subclass / custom Sequence — leave structurally intact.
        return value
    return value


def scrub_mapping(mapping: dict[str, Any], *, sensitive_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Return a *recursive* copy of ``mapping`` with sensitive content removed.

    Strings are run through :func:`scrub`. Nested ``dict`` values are
    recursed into with the same ``sensitive_keys`` set. Plain ``list`` /
    ``tuple`` values are reconstructed element-wise (uniformly, all the
    way down — see :func:`_scrub_value`); tuple subclasses pass through
    untouched.

    Keys whose name matches ``sensitive_keys`` (case-insensitive) — plus a
    built-in blocklist of credential-shaped names — are replaced outright
    regardless of value type, which handles structured PII like
    ``{"passport": "..."}`` where the value alone wouldn't match any
    generic pattern. Matching is *tokenised*: ``passportNumber`` and
    ``passenger_email`` are recognised because their key splits on
    camelCase / snake_case yield ``passport`` / ``email`` respectively.
    """
    sensitive = {k.lower() for k in sensitive_keys} | {
        'password', 'passport', 'passport_number', 'credit_card', 'card_number',
        'cvv', 'ssn', 'api_key', 'apikey', 'client_secret', 'access_token',
        'refresh_token', 'authorization', 'email', 'phone', 'phone_number',
    }
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        key_lower = key.lower()
        # 1) Exact-match path (backward compatible).
        # 2) Token path: split snake/camel-case, check against credential tokens.
        tokens = _split_key(key)
        if key_lower in sensitive or any(t in _SENSITIVE_TOKENS for t in tokens):
            out[key] = REDACTED
            continue
        out[key] = _scrub_value(value, sensitive)
    return out


def contains_pii(text: Any) -> bool:
    """Quick predicate used by tests and the logging guard."""
    if text is None:
        return False
    s = text if isinstance(text, str) else str(text)
    return any(p.search(s) for _, p in _PII_PATTERNS)
