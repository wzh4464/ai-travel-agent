"""Flight data source adapters.

The goal of this package is to decouple tool definitions from any particular
upstream provider. Every provider implements :class:`BaseFlightSource` and
returns results shaped by :mod:`agents.data_sources.normalizer`.
"""

from agents.data_sources.base import BaseFlightSource, RateLimiter, with_retries
from agents.data_sources.normalizer import Flight, FlightLeg, normalize_serpapi

__all__ = [
    'BaseFlightSource',
    'RateLimiter',
    'with_retries',
    'Flight',
    'FlightLeg',
    'normalize_serpapi',
    'SerpAPIFlightSource',
    'get_default_source',
]


def __getattr__(name):  # PEP 562 lazy imports
    """Defer the SerpAPI import until actually needed.

    This keeps the rest of the package (normalizer, rate limiter, retries)
    importable in environments where the optional ``serpapi`` dependency is
    not installed — useful for unit tests that exercise pure logic.
    """
    if name in ('SerpAPIFlightSource', 'get_default_source'):
        from agents.data_sources.serpapi_source import (  # noqa: WPS433
            SerpAPIFlightSource,
            get_default_source,
        )
        return {'SerpAPIFlightSource': SerpAPIFlightSource, 'get_default_source': get_default_source}[name]
    raise AttributeError(name)
