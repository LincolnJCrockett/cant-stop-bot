from cantstop.actions import legal_actions
from cantstop.bots import make_heuristic_bot
from cantstop.play import random_policy
from cantstop.tournament import (MatchResult, default_bots, format_table,
                                 play_match, play_one_game, round_robin,
                                 wilson_interval)


def test_play_one_game_is_reproducible_and_returns_a_seat():
    pols = [random_policy, random_policy]
    assert play_one_game(pols, 5) == play_one_game(pols, 5)
    assert play_one_game(pols, 5) in (0, 1)


def test_match_wins_add_up_and_is_reproducible():
    r1 = play_match(random_policy, random_policy, games=20, seed=1)
    r2 = play_match(random_policy, random_policy, games=20, seed=1)
    assert r1 == r2
    assert r1.wins_a + r1.wins_b == r1.games == 20


def test_policies_swap_seats_every_game():
    seats = []

    def spy(state, rng):
        seats.append(state.current_player)
        return legal_actions(state)[0]

    play_match(spy, random_policy, games=2, seed=0)
    assert set(seats) == {0, 1}          # sat in both seats across 2 games


def test_equal_bots_are_close_to_even():
    r = play_match(random_policy, random_policy, games=200, seed=0)
    lo, hi = r.interval_a
    assert lo <= 0.5 <= hi


def test_stronger_bot_wins_clearly_and_swapped_roles_agree():
    h = make_heuristic_bot(5)
    ab = play_match(h, random_policy, games=100, seed=0)
    ba = play_match(random_policy, h, games=100, seed=0)
    assert ab.win_rate_a > 0.8 and ba.win_rate_a < 0.2


def test_wilson_interval_known_values_and_shrinks_with_games():
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002
    lo2, hi2 = wilson_interval(500, 1000)
    assert (hi2 - lo2) < (hi - lo)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_round_robin_is_symmetric_and_complete():
    bots = {"a": random_policy, "b": make_heuristic_bot(5), "c": random_policy}
    res = round_robin(bots, games=10, seed=0)
    assert len(res) == 6                               # 3 pairs x 2 directions
    for (x, y), r in res.items():
        assert res[(y, x)] == MatchResult(r.wins_b, r.wins_a)


def test_table_lists_every_bot_and_average():
    bots = {"a": random_policy, "b": make_heuristic_bot(5)}
    text = format_table(list(bots), round_robin(bots, games=10))
    assert "a" in text and "b" in text and "average" in text
    assert len(default_bots()) >= 4
