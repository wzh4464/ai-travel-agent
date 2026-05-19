"""Unit tests for the region expansion and transit blacklist helpers."""

from __future__ import annotations

from agents.regions import (
    REGIONS,
    expand_region,
    expand_transit_blacklist,
)


class TestExpandRegion:
    def test_europe_has_major_hubs(self):
        codes = expand_region('europe')
        for hub in ['LHR', 'CDG', 'FRA', 'AMS', 'MAD', 'FCO', 'ZRH']:
            assert hub in codes

    def test_western_europe_is_subset_of_europe(self):
        assert set(expand_region('western_europe')).issubset(set(expand_region('europe_extended')))

    def test_cjk_europe_alias(self):
        assert expand_region('欧洲') == expand_region('europe')

    def test_cjk_northern_europe(self):
        codes = expand_region('北欧')
        assert 'CPH' in codes and 'ARN' in codes and 'HEL' in codes

    def test_case_insensitive(self):
        assert expand_region('EUROPE') == expand_region('europe')

    def test_whitespace_and_dash(self):
        assert expand_region('western europe') == expand_region('western_europe')
        assert expand_region('western-europe') == expand_region('western_europe')

    def test_unknown_region_returns_empty(self):
        assert expand_region('atlantis') == []

    def test_empty_input(self):
        assert expand_region('') == []
        assert expand_region(None) == []

    def test_all_declared_regions_are_non_empty(self):
        for name, codes in REGIONS.items():
            assert len(codes) >= 3, f'region {name} looks too small'


class TestTransitBlacklist:
    def test_middle_east_covers_gulf_hubs(self):
        codes = expand_transit_blacklist('middle_east')
        for hub in ['DXB', 'DOH', 'AUH', 'RUH', 'KWI', 'BAH']:
            assert hub in codes

    def test_middle_east_excludes_istanbul_by_default(self):
        assert 'IST' not in expand_transit_blacklist('middle_east')

    def test_middle_east_strict_includes_istanbul(self):
        codes = expand_transit_blacklist('middle_east_strict')
        assert 'IST' in codes
        assert 'DXB' in codes  # still includes the gulf hubs

    def test_cjk_middle_east(self):
        assert expand_transit_blacklist('中东') == expand_transit_blacklist('middle_east')

    def test_cjk_strict_variant(self):
        assert expand_transit_blacklist('中东严格') == expand_transit_blacklist('middle_east_strict')

    def test_raw_iata_code_passthrough(self):
        assert expand_transit_blacklist('DXB') == {'DXB'}

    def test_empty_input(self):
        assert expand_transit_blacklist('') == set()
