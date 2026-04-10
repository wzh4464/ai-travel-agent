"""Shared fixtures and path setup for the test suite.

Running ``pytest`` from the repo root works without installation because
this file prepends the project root to ``sys.path``. All network-backed
tests are gated on HTTP mocks; no test should ever hit a real provider.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIXTURES_DIR = pathlib.Path(__file__).parent / 'fixtures'


def _load_json(name: str) -> dict:
    with (FIXTURES_DIR / name).open('r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='session')
def serpapi_flights_fixture() -> dict:
    return _load_json('serpapi_flights.json')


@pytest.fixture(scope='session')
def amadeus_offers_fixture() -> dict:
    return _load_json('amadeus_offers.json')


@pytest.fixture(scope='session')
def amadeus_token_fixture() -> dict:
    return _load_json('amadeus_token.json')


@pytest.fixture(scope='session')
def kiwi_search_fixture() -> dict:
    return _load_json('kiwi_search.json')


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Make sure a stray ``SERPAPI_API_KEY`` on the developer's laptop cannot
    reach a test. Every test starts with a clean slate; individual tests can
    set the keys they need."""
    for key in (
        'SERPAPI_API_KEY',
        'AMADEUS_CLIENT_ID',
        'AMADEUS_CLIENT_SECRET',
        'AMADEUS_BASE_URL',
        'TEQUILA_API_KEY',
        'KIWI_API_KEY',
        'KIWI_BASE_URL',
        'FLIGHT_SOURCES',
    ):
        monkeypatch.delenv(key, raising=False)
    # Reset aggregator singleton between tests so env changes take effect.
    try:
        import agents.data_sources.aggregator as _agg  # noqa: WPS433
        _agg._default_aggregator = None
    except ImportError:
        pass
