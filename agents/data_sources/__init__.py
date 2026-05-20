"""Flight data source adapters.

The goal of this package is to decouple tool definitions from any particular
upstream provider. Every provider implements :class:`BaseFlightSource` and
returns results shaped by :mod:`agents.data_sources.normalizer`.

The :class:`AggregatedFlightSource` fans a single search request out to
every configured provider in parallel and merges the results.
"""

from agents.data_sources.aggregator import (
    AggregatedFlightSource,
    build_default_aggregator,
    get_default_aggregator,
)
from agents.data_sources.base import BaseFlightSource, RateLimiter, with_retries
from agents.data_sources.normalizer import (
    Flight,
    FlightLeg,
    normalize_amadeus,
    normalize_kiwi,
    normalize_serpapi,
)

__all__ = [
    'BaseFlightSource',
    'RateLimiter',
    'with_retries',
    'Flight',
    'FlightLeg',
    'normalize_serpapi',
    'normalize_amadeus',
    'normalize_kiwi',
    'AggregatedFlightSource',
    'build_default_aggregator',
    'get_default_aggregator',
    'SerpAPIFlightSource',
    'AmadeusFlightSource',
    'KiwiFlightSource',
    'get_default_source',
]


def __getattr__(name):  # PEP 562 lazy imports
    """Defer optional imports so missing SDKs do not break the package."""
    if name == 'SerpAPIFlightSource' or name == 'get_default_source':
        from agents.data_sources.serpapi_source import (  # noqa: WPS433
            SerpAPIFlightSource,
            get_default_source,
        )
        return {
            'SerpAPIFlightSource': SerpAPIFlightSource,
            'get_default_source': get_default_source,
        }[name]
    if name == 'AmadeusFlightSource':
        from agents.data_sources.amadeus_source import AmadeusFlightSource  # noqa: WPS433
        return AmadeusFlightSource
    if name == 'KiwiFlightSource':
        from agents.data_sources.kiwi_source import KiwiFlightSource  # noqa: WPS433
        return KiwiFlightSource
    raise AttributeError(name)
