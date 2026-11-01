from dataclasses import replace

import numpy as np
import pytest

from cantstop.actions import CONTINUE, STOP, Action, Kind, legal_actions
from cantstop.engine import apply, needs_roll, roll, roll_until_decision, step
from cantstop.state import COLUMN_LENGTHS, new_game


def play(*cols):
    return Action(Kind.PLAY, tuple(cols))


def progress_with(player_progress):
    """progress tuple for 2 players; player 0 given as {column: steps}."""
    p0 = [0] * 11
    for c, v in player_progress.items():
        p0[c - 2] = v
    return (tuple(p0), (0,) * 11)


def test_play_places_new_runners_from_permanent_progress():
    s = replace(new_game(), dice=(1, 2, 3, 4), progress=progress_with({3: 2}))
    s = apply(s, play(3, 7))
    assert s.runners == ((3, 3), (7, 1))
    assert s.dice is None and s.rolled


def test_double_sum_moves_two_steps_and_caps_at_top():
    s = replace(new_game(), dice=(1, 1, 1, 1), runners=((2, 2),))
    s = apply(s, play(2, 2))
    assert s.runners == ((2, 3),)          # length 3, advanced only 1


def test_illegal_action_rejected():
    s = replace(new_game(), dice=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        apply(s, play(12))


def test_stop_banks_progress_and_passes_turn():
    s = replace(new_game(), runners=((6, 4), (8, 2)), rolled=True)
    s = apply(s, STOP)
    assert s.progress[0][6 - 2] == 4 and s.progress[0][8 - 2] == 2
    assert s.runners == () and not s.rolled
    assert s.current_player == 1
    assert s.owners == (None,) * 11


def test_stop_at_top_captures_column():
    s = replace(new_game(), runners=((2, 3), (7, 1)), rolled=True)
    s = apply(s, STOP)
    assert s.owners[0] == 0                # column 2 captured by player 0
    assert s.owners[7 - 2] is None


def test_can_capture_multiple_columns_in_one_turn():
    s = replace(new_game(), runners=((2, 3), (12, 3)), rolled=True)
    s = apply(s, STOP)
    assert s.owners[0] == 0 and s.owners[10] == 0


def test_third_capture_wins_immediately():
    owners = (0, 0) + (None,) * 9
    s = replace(new_game(), owners=owners, runners=((12, 3),), rolled=True)
    s = apply(s, STOP)
    assert s.winner == 0 and s.is_over
    assert s.current_player == 0           # turn does not pass
    assert legal_actions(s) == []


def test_continue_means_a_roll_is_due():
    s = replace(new_game(), runners=((6, 1),), rolled=True)
    s = apply(s, CONTINUE)
    assert needs_roll(s) and s.runners == ((6, 1),)


def test_bust_discards_runners_and_passes_turn():
    # 3 runners on 2, 11, 12: almost no roll hits them; search for a bust.
    base = replace(new_game(), runners=((2, 0), (11, 0), (12, 0)))
    for seed in range(200):
        s = roll(base, np.random.default_rng(seed))
        if s.dice is None:                 # bust happened
            assert s.runners == ()
            assert s.current_player == 1
            assert s.progress == base.progress
            return
    pytest.fail("no bust found in 200 seeds")


def test_bust_captures_nothing():
    base = replace(new_game(), runners=((2, 3), (11, 0), (12, 0)))
    for seed in range(200):
        s = roll(base, np.random.default_rng(seed))
        if s.dice is None:
            assert s.owners == (None,) * 11
            return
    pytest.fail("no bust found")


def test_roll_until_decision_stops_at_a_decision():
    s = roll_until_decision(new_game(), np.random.default_rng(0))
    assert s.dice is not None and legal_actions(s)


def test_random_games_finish_and_keep_invariants():
    rng = np.random.default_rng(123)
    for game in range(100):
        s = roll_until_decision(new_game(), rng)
        steps = 0
        while not s.is_over:
            actions = legal_actions(s)
            assert actions
            s = step(s, actions[rng.integers(len(actions))], rng)
            steps += 1
            assert steps < 20000, "game did not terminate"
            assert len(s.runners) <= 3
            for player_progress in s.progress:
                for col, v in zip(COLUMN_LENGTHS, player_progress):
                    assert 0 <= v <= COLUMN_LENGTHS[col]
        assert s.owners.count(s.winner) >= 3


def test_same_seed_same_game():
    def run(seed):
        rng = np.random.default_rng(seed)
        s = roll_until_decision(new_game(), rng)
        while not s.is_over:
            acts = legal_actions(s)
            s = step(s, acts[rng.integers(len(acts))], rng)
        return s
    assert run(7) == run(7)
