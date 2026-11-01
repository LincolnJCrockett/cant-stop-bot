"""Tests for the column-progress bot (written by Claude)."""
from dataclasses import replace

import numpy as np

from cantstop.actions import CONTINUE, STOP, legal_actions
from cantstop.bots import (column_progress_this_turn,
                           make_stop_after_column_progress)
from cantstop.play import play_game, random_policy
from cantstop.state import new_game


def deciding(runners, **changes):
    """A stop/continue decision point with the given runners."""
    return replace(new_game(), runners=runners, rolled=True, **changes)


def test_progress_is_fraction_of_column_gained():
    assert column_progress_this_turn(new_game()) == 0
    # 3 steps on column 2 (length 3) = a whole column.
    assert column_progress_this_turn(deciding(((2, 3),))) == 1.0
    # 13 steps on column 7 (length 13) = a whole column too.
    assert column_progress_this_turn(deciding(((7, 13),))) == 1.0


def test_progress_sums_over_runners():
    # col 2: 1/3 + col 12: 1/3 + col 7: 13/13*... use 6.5/13 not possible; use 2/3.
    s = deciding(((2, 1), (12, 1), (7, 13)))
    assert abs(column_progress_this_turn(s) - (1/3 + 1/3 + 1.0)) < 1e-9


def test_progress_ignores_banked_steps():
    p0 = [0] * 11
    p0[7 - 2] = 10                       # 10 banked on column 7
    s = deciding(((7, 12),), progress=(tuple(p0), (0,) * 11))
    # Gained only 2 steps this turn: 2/13.
    assert abs(column_progress_this_turn(s) - 2 / 13) < 1e-9


def test_stops_at_threshold_and_continues_below():
    bot = make_stop_after_column_progress(0.5)
    rng = np.random.default_rng(0)
    # col 12: 2/3 = 0.67 >= 0.5 -> stop
    assert bot(deciding(((12, 2),)), rng) == STOP
    # col 7: 6/13 = 0.46 < 0.5 -> continue
    assert bot(deciding(((7, 6),)), rng) == CONTINUE
    # col 7: 7/13 = 0.54 >= 0.5 -> stop
    assert bot(deciding(((7, 7),)), rng) == STOP


def test_short_column_steps_count_for_more_than_long_column_steps():
    bot = make_stop_after_column_progress(0.5)
    rng = np.random.default_rng(0)
    assert bot(deciding(((2, 2),)), rng) == STOP        # 2/3
    assert bot(deciding(((7, 2),)), rng) == CONTINUE    # 2/13


def test_picks_a_legal_play_at_random_when_dice_pending():
    bot = make_stop_after_column_progress(0.5)
    s = replace(new_game(), dice=(1, 2, 3, 4))
    legal = legal_actions(s)
    chosen = {bot(s, np.random.default_rng(seed)) for seed in range(60)}
    assert chosen <= set(legal)
    assert len(chosen) == len(legal)          # all options get picked


def test_bot_only_ever_returns_legal_actions():
    rng = np.random.default_rng(1)
    bot = make_stop_after_column_progress(0.5)
    from cantstop.engine import roll_until_decision, step
    for _ in range(10):
        s = roll_until_decision(new_game(), rng)
        while not s.is_over:
            a = bot(s, rng)
            assert a in legal_actions(s)
            s = step(s, a, rng)


def test_beats_random_player_clearly():
    bot = make_stop_after_column_progress(0.5)
    wins = 0
    games = 100
    for g in range(games):
        seat = g % 2                                  # alternate who goes first
        policies = [bot, random_policy] if seat == 0 else [random_policy, bot]
        final = play_game(policies, seed=g, print_fn=lambda *_: None,
                          verbose_bots=False)
        wins += final.winner == seat
    assert wins / games > 0.75
