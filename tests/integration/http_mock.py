"""Tiny stub for ``urllib.request.urlopen``.

The Amadeus and Kiwi adapters deliberately use only the standard library so
they can be tested without pulling in ``requests``/``responses``. This
helper gives each test a programmable queue of responses keyed by URL
prefix.

Usage::

    with mock_urlopen() as mock:
        mock.add('POST', 'https://test.api.amadeus.com/v1/security', token_json)
        mock.add('GET', 'https://test.api.amadeus.com/v2/shopping', offers_json)
        ...  # exercise code under test
        assert mock.requests[-1].full_url.startswith('https://test.api.amadeus.com/v2')
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch
from urllib.error import HTTPError


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FlakyURLOpen:
    """Programmable urllib stub.

    Tests add ``(method, url_prefix, handler)`` entries in order, then call
    the module under test. Each call consumes the first entry whose prefix
    matches the incoming request. ``handler`` can be:

    * a dict  → serialised to JSON with status 200
    * bytes   → returned raw
    * an Exception → raised
    * a callable(request) → anything of the above
    """

    def __init__(self) -> None:
        self._queue: list[tuple[str, str, Any]] = []
        self.requests: list[Any] = []

    def add(self, method: str, url_prefix: str, handler: Any) -> None:
        self._queue.append((method.upper(), url_prefix, handler))

    def __call__(self, request, *args, **kwargs):  # mimics urllib.request.urlopen
        self.requests.append(request)
        method = (getattr(request, 'get_method', lambda: 'GET')() or 'GET').upper()
        url = getattr(request, 'full_url', str(request))
        for i, (m, prefix, handler) in enumerate(self._queue):
            if m == method and url.startswith(prefix):
                del self._queue[i]
                return self._materialise(handler, request)
        raise AssertionError(
            f'Unexpected HTTP {method} to {url}; queue={[ (m, p) for m, p, _ in self._queue ]}'
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _materialise(handler: Any, request: Any) -> _FakeResponse:
        if callable(handler) and not isinstance(handler, Exception):
            handler = handler(request)
        if isinstance(handler, Exception):
            raise handler
        if isinstance(handler, dict):
            body = json.dumps(handler).encode('utf-8')
            return _FakeResponse(body, status=200)
        if isinstance(handler, (bytes, bytearray)):
            return _FakeResponse(bytes(handler), status=200)
        if isinstance(handler, str):
            return _FakeResponse(handler.encode('utf-8'), status=200)
        raise TypeError(f'Unsupported handler: {type(handler)!r}')


def make_http_error(url: str, code: int, body: bytes = b'', headers: dict | None = None) -> HTTPError:
    fp = io.BytesIO(body)
    return HTTPError(url, code, 'error', headers or {}, fp)


@contextmanager
def mock_urlopen(*modules: str) -> Iterator[FlakyURLOpen]:
    """Patch ``urllib.request.urlopen`` across one or more modules.

    Each entry is a fully-qualified module name whose local ``urlopen``
    reference should be replaced. The Amadeus and Kiwi adapters both use
    ``import urllib.request`` and call ``urllib.request.urlopen`` via
    attribute access, so patching the module-level attribute once is
    enough — but when modules do ``from urllib.request import urlopen``
    we need the per-module override.
    """
    mock = FlakyURLOpen()
    targets = ['urllib.request.urlopen']
    # Cover both styles: ``import urllib.request`` (patch the module's
    # urllib.request.urlopen reference) and ``from urllib.request import
    # urlopen`` (patch the module-local ``urlopen`` symbol).
    for m in modules:
        targets.append(f'{m}.urllib.request.urlopen')
        targets.append(f'{m}.urlopen')
    patchers = []
    for t in targets:
        try:
            p = patch(t, mock)
            p.start()
            patchers.append(p)
        except (AttributeError, ModuleNotFoundError):
            continue
    try:
        yield mock
    finally:
        for p in patchers:
            p.stop()
