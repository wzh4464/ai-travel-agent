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


def scrub_mapping(mapping: dict[str, Any], *, sensitive_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Return a shallow copy of ``mapping`` with known-sensitive keys and
    every string value scrubbed.

    Keys whose name matches ``sensitive_keys`` (case-insensitive) are
    replaced outright regardless of value type — this handles structured
    PII like ``{"passport": "..."}`` where the value wouldn't match any
    generic pattern.
    """
    sensitive = {k.lower() for k in sensitive_keys} | {
        'password', 'passport', 'passport_number', 'credit_card', 'card_number',
        'cvv', 'ssn', 'api_key', 'apikey', 'client_secret', 'access_token',
        'refresh_token', 'authorization', 'email', 'phone', 'phone_number',
    }
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if key.lower() in sensitive:
            out[key] = REDACTED
            continue
        if isinstance(value, str):
            out[key] = scrub(value)
        elif isinstance(value, dict):
            out[key] = scrub_mapping(value, sensitive_keys=sensitive_keys)
        elif isinstance(value, (list, tuple)):
            out[key] = type(value)(
                scrub(v) if isinstance(v, str)
                else scrub_mapping(v, sensitive_keys=sensitive_keys) if isinstance(v, dict)
                else v
                for v in value
            )
        else:
            out[key] = value
    return out


def contains_pii(text: Any) -> bool:
    """Quick predicate used by tests and the logging guard."""
    if text is None:
        return False
    s = text if isinstance(text, str) else str(text)
    return any(p.search(s) for _, p in _PII_PATTERNS)
