from dataclasses import replace

import numpy as np

from cantstop.actions import CONTINUE, STOP, Action, Kind, legal_actions
from cantstop.bots import (choose_play, make_heuristic_bot, make_stop_after_n,
                           play_score)
from cantstop.engine import roll_until_decision, step
from cantstop.play import play_game, random_policy
from cantstop.state import new_game


def play(*cols):
    return Action(Kind.PLAY, tuple(cols))


def pending(dice, **changes):
    return replace(new_game(), dice=dice, **changes)


def deciding(runners, **changes):
    return replace(new_game(), runners=runners, rolled=True, **changes)


# --- play_score -----------------------------------------------------------
def test_common_column_scores_higher_than_rare_column():
    s = new_game()
    assert play_score(s, play(7)) > play_score(s, play(2))
    assert play_score(s, play(6)) > play_score(s, play(3))


def test_two_sums_score_higher_than_one():
    s = new_game()
    assert play_score(s, play(6, 8)) > play_score(s, play(6))


def test_existing_runner_beats_new_runner_on_same_column():
    with_runner = replace(new_game(), runners=((4, 1),))
    assert play_score(with_runner, play(4)) > play_score(new_game(), play(4))


def test_completing_a_column_gets_a_bonus():
    near_top = replace(new_game(), runners=((12, 2),))      # length 3
    mid = replace(new_game(), runners=((12, 1),))
    assert play_score(near_top, play(12)) > play_score(mid, play(12))


def test_capture_bonus_uses_banked_progress_for_new_runner():
    p0 = [0] * 11
    p0[2 - 2] = 2                                   # column 2: 2 of 3 banked
    s = replace(new_game(), progress=(tuple(p0), (0,) * 11))
    fresh = new_game()
    assert play_score(s, play(2)) > play_score(fresh, play(2))


# --- choose_play ----------------------------------------------------------
def test_prefers_pairing_with_existing_runner():
    # Dice 1,2,3,4 -> (3,7), (4,6), (5,5); runner already on 4.
    s = pending((1, 2, 3, 4), runners=((4, 1),))
    for seed in range(20):
        assert choose_play(s, np.random.default_rng(seed)) == play(4, 6)


def test_prefers_common_columns_when_nothing_else_differs():
    # Dice 1,1,6,6 -> (2,12) or (7,7).
    s = pending((1, 1, 6, 6))
    for seed in range(20):
        assert choose_play(s, np.random.default_rng(seed)) == play(7, 7)


def test_takes_the_capture_when_available():
    # Runner on 5 at 8 of 9. Dice 1,2,3,4 -> (3,7), (4,6), (5,5).
    s = pending((1, 2, 3, 4), runners=((5, 8),))
    for seed in range(20):
        assert choose_play(s, np.random.default_rng(seed)) == play(5, 5)


def test_ties_are_broken_randomly():
    # Dice 1,2,3,4 on an empty board: all three pairings tie on score.
    s = pending((1, 2, 3, 4))
    chosen = {choose_play(s, np.random.default_rng(seed)) for seed in range(60)}
    assert len(chosen) > 1


# --- the full bot ---------------------------------------------------------
def test_stop_rule_matches_stop_after_n():
    bot = make_heuristic_bot(5)
    rng = np.random.default_rng(0)
    assert bot(deciding(((7, 4),)), rng) == CONTINUE
    assert bot(deciding(((7, 5),)), rng) == STOP
    assert bot(deciding(((6, 2), (8, 3))), rng) == STOP


def test_counts_only_steps_gained_this_turn():
    bot = make_heuristic_bot(3)
    p0 = [0] * 11
    p0[7 - 2] = 10
    prog = (tuple(p0), (0,) * 11)
    rng = np.random.default_rng(0)
    assert bot(deciding(((7, 12),), progress=prog), rng) == CONTINUE
    assert bot(deciding(((7, 13),), progress=prog), rng) == STOP


def test_only_ever_returns_legal_actions():
    rng = np.random.default_rng(3)
    bot = make_heuristic_bot(5)
    for _ in range(10):
        s = roll_until_decision(new_game(), rng)
        while not s.is_over:
            a = bot(s, rng)
            assert a in legal_actions(s)
            s = step(s, a, rng)


def win_rate(bot, opponent, games=100):
    wins = 0
    for g in range(games):
        seat = g % 2
        policies = [bot, opponent] if seat == 0 else [opponent, bot]
        final = play_game(policies, seed=g, print_fn=lambda *_: None,
                          verbose_bots=False)
        wins += final.winner == seat
    return wins / games


def test_beats_random_player_clearly():
    assert win_rate(make_heuristic_bot(5), random_policy) > 0.8


def test_smart_pairing_beats_random_pairing_with_same_stop_rule():
    assert win_rate(make_heuristic_bot(5), make_stop_after_n(5)) > 0.6
