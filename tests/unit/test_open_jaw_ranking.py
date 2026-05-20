"""Tests for the pure-data open-jaw ranking and rendering."""

from __future__ import annotations

from agents.presentation.itinerary import (
    format_open_jaw_combinations,
    rank_open_jaw_combinations,
)


def _f(fid, price, legs, *, currency='HKD', duration=720):
    return {
        'flight_id': fid,
        'price': price,
        'currency': currency,
        'total_duration_minutes': duration,
        'stops': max(0, len(legs) - 1),
        'legs': [
            {'departure_airport': a, 'arrival_airport': b, 'airline': 'XX',
             'flight_number': 'XX1', 'departure_time': '', 'arrival_time': '',
             'duration_minutes': 300, 'aircraft': '', 'cabin_class': 'economy'}
            for a, b in legs
        ],
        'provider': 'stub',
    }


class TestRankOpenJawCombinations:
    def test_cheapest_open_jaw_wins(self):
        outbound = {
            'CDG': [_f('ob1', 3500, [('HKG', 'CDG')])],
            'FCO': [_f('ob2', 3200, [('HKG', 'FCO')])],
            'LHR': [_f('ob3', 4100, [('HKG', 'LHR')])],
        }
        returns = {
            'CDG': [_f('r1', 3800, [('CDG', 'HKG')])],
            'FCO': [_f('r2', 4200, [('FCO', 'HKG')])],
            'LHR': [_f('r3', 3400, [('LHR', 'HKG')])],
        }
        ranked = rank_open_jaw_combinations(outbound, returns, top_n=3)
        top = ranked[0]
        # Cheapest outbound = HKG→FCO 3200
        # Cheapest return   = LHR→HKG 3400
        # Total             = 6600
        assert top['total_price'] == 6600
        assert top['entry_city'] == 'FCO'
        assert top['exit_city'] == 'LHR'
        assert top['open_jaw'] is True

    def test_same_city_constraint(self):
        outbound = {
            'CDG': [_f('ob1', 3500, [('HKG', 'CDG')])],
            'FCO': [_f('ob2', 3200, [('HKG', 'FCO')])],
        }
        returns = {
            'CDG': [_f('r1', 3800, [('CDG', 'HKG')])],
            'FCO': [_f('r2', 4200, [('FCO', 'HKG')])],
        }
        ranked = rank_open_jaw_combinations(outbound, returns, same_city=True, top_n=5)
        assert all(c['entry_city'] == c['exit_city'] for c in ranked)
        assert all(c['open_jaw'] is False for c in ranked)
        # Cheapest same-city = CDG round-trip at 3500+3800 = 7300
        assert ranked[0]['total_price'] == 7300
        assert ranked[0]['entry_city'] == 'CDG'

    def test_banned_transit_removes_flights_before_ranking(self):
        outbound = {
            # Cheapest to CDG routes via DXB — should be removed
            'CDG': [
                _f('ob_dxb', 2800, [('HKG', 'DXB'), ('DXB', 'CDG')]),
                _f('ob_direct', 3500, [('HKG', 'CDG')]),
            ],
        }
        returns = {
            'CDG': [_f('r1', 3800, [('CDG', 'HKG')])],
        }
        ranked = rank_open_jaw_combinations(
            outbound, returns, banned_transit={'DXB'}, same_city=True, top_n=5
        )
        assert len(ranked) == 1
        # Total should use the direct outbound, not the DXB one
        assert ranked[0]['total_price'] == 3500 + 3800
        # And the outbound on the combination should not touch DXB
        ob_legs = ranked[0]['outbound']['legs']
        assert all(leg['arrival_airport'] != 'DXB' for leg in ob_legs[:-1])

    def test_max_price_caps_results(self):
        outbound = {'CDG': [_f('ob', 5000, [('HKG', 'CDG')])]}
        returns = {'CDG': [_f('r', 5000, [('CDG', 'HKG')])]}
        ranked = rank_open_jaw_combinations(outbound, returns, max_price=9000, same_city=True)
        assert ranked == []  # 10000 > 9000

    def test_top_n_limits_output(self):
        outbound = {f'C{i}': [_f(f'ob{i}', 3000 + i, [('HKG', f'C{i}')])] for i in range(20)}
        returns = {f'C{i}': [_f(f'r{i}', 3000 + i, [(f'C{i}', 'HKG')])] for i in range(20)}
        ranked = rank_open_jaw_combinations(outbound, returns, same_city=True, top_n=5)
        assert len(ranked) == 5
        prices = [c['total_price'] for c in ranked]
        assert prices == sorted(prices)

    def test_per_city_candidates_limits_expansion(self):
        # If we allow unlimited candidates per city, 10 outbounds × 10 returns = 100
        # With per_city_candidates=2, we get 2 × 2 × 1 city = 4
        outbound = {'CDG': [_f(f'ob{i}', 1000 + i, [('HKG', 'CDG')]) for i in range(10)]}
        returns = {'CDG': [_f(f'r{i}', 1000 + i, [('CDG', 'HKG')]) for i in range(10)]}
        ranked = rank_open_jaw_combinations(
            outbound, returns, same_city=True, top_n=100, per_city_candidates=2,
        )
        assert len(ranked) == 4

    def test_empty_input(self):
        assert rank_open_jaw_combinations({}, {}) == []

    def test_outbound_city_with_no_valid_flights_is_skipped(self):
        outbound = {
            'CDG': [_f('ob_dxb', 2000, [('HKG', 'DXB'), ('DXB', 'CDG')])],
            'FCO': [_f('ob_fco', 3200, [('HKG', 'FCO')])],
        }
        returns = {
            'CDG': [_f('r_cdg', 3800, [('CDG', 'HKG')])],
            'FCO': [_f('r_fco', 4000, [('FCO', 'HKG')])],
        }
        ranked = rank_open_jaw_combinations(
            outbound, returns, banned_transit={'DXB'}, same_city=True,
        )
        # CDG outbound was filtered out, so only FCO survives
        assert len(ranked) == 1
        assert ranked[0]['entry_city'] == 'FCO'


class TestFormatOpenJawCombinations:
    def test_empty_returns_hint(self):
        out = format_open_jaw_combinations([])
        assert 'No itineraries' in out

    def test_renders_cards_with_totals(self):
        outbound = {'CDG': [_f('ob', 3500, [('HKG', 'CDG')])]}
        returns = {'CDG': [_f('r', 3800, [('CDG', 'HKG')])]}
        ranked = rank_open_jaw_combinations(outbound, returns, same_city=True)
        rendered = format_open_jaw_combinations(ranked)
        assert 'Option 1' in rendered
        assert 'CDG' in rendered
        assert 'HKD 7,300' in rendered
        assert 'Outbound' in rendered
        assert 'Return' in rendered

    def test_open_jaw_badge(self):
        outbound = {'CDG': [_f('ob', 3500, [('HKG', 'CDG')])]}
        returns = {'FCO': [_f('r', 3800, [('FCO', 'HKG')])]}
        ranked = rank_open_jaw_combinations(outbound, returns, top_n=1)
        rendered = format_open_jaw_combinations(ranked)
        assert 'open-jaw' in rendered

    def test_shows_transit_airport_in_stops(self):
        outbound = {'CDG': [_f('ob', 3500, [('HKG', 'AMS'), ('AMS', 'CDG')])]}
        returns = {'CDG': [_f('r', 3800, [('CDG', 'HKG')])]}
        ranked = rank_open_jaw_combinations(outbound, returns, same_city=True)
        rendered = format_open_jaw_combinations(ranked)
        assert 'AMS' in rendered
