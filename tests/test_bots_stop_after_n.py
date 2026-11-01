"""Tests for YOUR stop-after-N-steps bot.

Write `make_stop_after_n(n)` in src/cantstop/bots.py. It should return a
policy (state, rng) -> Action that
  - picks a legal PLAY at random when dice are pending
  - otherwise returns STOP if the raw steps gained this turn >= n,
    else CONTINUE.
These tests skip until the function exists.
"""
from dataclasses import replace

import numpy as np
import pytest

import cantstop.bots as bots
from cantstop.actions import CONTINUE, STOP, legal_actions
from cantstop.engine import roll_until_decision, step
from cantstop.play import play_game, random_policy
from cantstop.state import new_game

make_stop_after_n = getattr(bots, "make_stop_after_n", None)
pytestmark = pytest.mark.skipif(
    make_stop_after_n is None, reason="make_stop_after_n not written yet")


def deciding(runners, **changes):
    return replace(new_game(), runners=runners, rolled=True, **changes)


def test_stops_at_n_and_continues_below():
    bot = make_stop_after_n(5)
    rng = np.random.default_rng(0)
    assert bot(deciding(((7, 4),)), rng) == CONTINUE     # 4 steps
    assert bot(deciding(((7, 5),)), rng) == STOP         # 5 steps
    assert bot(deciding(((7, 9),)), rng) == STOP         # more than 5


def test_steps_sum_across_runners():
    bot = make_stop_after_n(5)
    rng = np.random.default_rng(0)
    assert bot(deciding(((6, 2), (8, 2))), rng) == CONTINUE   # 4 total
    assert bot(deciding(((6, 2), (8, 3))), rng) == STOP       # 5 total


def test_counts_only_steps_gained_this_turn():
    bot = make_stop_after_n(3)
    rng = np.random.default_rng(0)
    p0 = [0] * 11
    p0[7 - 2] = 10                                   # banked 10 on column 7
    prog = (tuple(p0), (0,) * 11)
    assert bot(deciding(((7, 12),), progress=prog), rng) == CONTINUE  # gained 2
    assert bot(deciding(((7, 13),), progress=prog), rng) == STOP      # gained 3


def test_steps_not_weighted_by_column_length():
    bot = make_stop_after_n(2)
    rng = np.random.default_rng(0)
    assert bot(deciding(((2, 2),)), rng) == STOP         # short column
    assert bot(deciding(((7, 2),)), rng) == STOP         # long column, same


def test_picks_a_legal_play_at_random_when_dice_pending():
    bot = make_stop_after_n(5)
    s = replace(new_game(), dice=(1, 2, 3, 4))
    legal = legal_actions(s)
    chosen = {bot(s, np.random.default_rng(seed)) for seed in range(60)}
    assert chosen <= set(legal) and len(chosen) == len(legal)


def test_bot_only_ever_returns_legal_actions():
    rng = np.random.default_rng(2)
    bot = make_stop_after_n(5)
    for _ in range(10):
        s = roll_until_decision(new_game(), rng)
        while not s.is_over:
            a = bot(s, rng)
            assert a in legal_actions(s)
            s = step(s, a, rng)


def test_a_sensible_n_beats_random():
    bot = make_stop_after_n(6)
    wins, games = 0, 100
    for g in range(games):
        seat = g % 2
        policies = [bot, random_policy] if seat == 0 else [random_policy, bot]
        final = play_game(policies, seed=g, print_fn=lambda *_: None,
                          verbose_bots=False)
        wins += final.winner == seat
    assert wins / games > 0.6
