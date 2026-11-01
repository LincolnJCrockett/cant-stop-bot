from dataclasses import replace

import numpy as np

from cantstop.actions import CONTINUE, STOP, legal_actions
from cantstop.engine import roll_until_decision, step
from cantstop.env import (ALL_ACTIONS, NUM_ACTIONS, action_to_index,
                          index_to_action, legal_mask)
from cantstop.state import new_game


def test_action_space_size_and_uniqueness():
    assert NUM_ACTIONS == 79
    assert len(set(ALL_ACTIONS)) == 79


def test_index_roundtrip():
    for i in range(NUM_ACTIONS):
        assert action_to_index(index_to_action(i)) == i


def test_mask_matches_legal_actions():
    s = replace(new_game(), dice=(1, 2, 3, 4))
    mask = legal_mask(s)
    assert mask.dtype == bool and mask.shape == (79,)
    legal = legal_actions(s)
    assert mask.sum() == len(legal)
    for a in legal:
        assert mask[action_to_index(a)]


def test_mask_for_stop_continue_phase():
    s = replace(new_game(), rolled=True, runners=((6, 1),))
    mask = legal_mask(s)
    assert mask[action_to_index(STOP)] and mask[action_to_index(CONTINUE)]
    assert mask.sum() == 2


def test_every_legal_action_in_random_games_is_in_the_space():
    rng = np.random.default_rng(0)
    for _ in range(30):
        s = roll_until_decision(new_game(), rng)
        while not s.is_over:
            mask = legal_mask(s)           # KeyError here = missing action
            assert mask.sum() == len(legal_actions(s)) >= 1
            choices = np.flatnonzero(mask)
            s = step(s, index_to_action(int(rng.choice(choices))), rng)


# --- observation encoding ---
from cantstop.env import OBS_SIZE, encode_observation  # noqa: E402


def block(obs, i):
    """The i-th block of 11 per-column values."""
    return obs[11 * i: 11 * (i + 1)]


def test_observation_shape_dtype_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = roll_until_decision(new_game(), rng)
        while not s.is_over:
            obs = encode_observation(s)
            assert obs.shape == (OBS_SIZE,) == (68,)
            assert obs.dtype == np.float32
            assert obs.min() >= 0.0 and obs.max() <= 1.0
            acts = legal_actions(s)
            s = step(s, acts[rng.integers(len(acts))], rng)


def test_observation_values():
    p0 = [0] * 11
    p0[6 - 2] = 5           # column 6 (length 11): 5 steps banked
    s = replace(
        new_game(),
        progress=(tuple(p0), (0,) * 11),
        owners=(None, None, None, None, None, None, None, None, None, None, 1),
        runners=((7, 13),), rolled=True,
    )
    obs = encode_observation(s)
    assert block(obs, 0)[6 - 2] == np.float32(5 / 11)       # my progress
    assert block(obs, 2)[7 - 2] == 1.0                      # runner at top of 7
    assert block(obs, 3)[7 - 2] == 1.0 and block(obs, 3).sum() == 1
    assert block(obs, 5)[12 - 2] == 1.0                     # opp captured 12
    assert block(obs, 4).sum() == 0
    assert obs[66] == np.float32(2 / 3)                     # 2 runners free
    assert obs[67] == 0.0                                   # stop/continue phase


def test_phase_flag_when_dice_pending():
    s = replace(new_game(), dice=(1, 2, 3, 4))
    assert encode_observation(s)[67] == 1.0


def test_observation_is_from_current_players_perspective():
    p0 = [0] * 11
    p0[4] = 3
    a = replace(new_game(), progress=(tuple(p0), (0,) * 11), current_player=0)
    # Same position with the seats swapped and player 1 to move.
    b = replace(new_game(), progress=((0,) * 11, tuple(p0)), current_player=1)
    assert np.array_equal(encode_observation(a), encode_observation(b))


# --- environment wrapper ---
import pytest  # noqa: E402

from cantstop.env import CantStopEnv  # noqa: E402


def random_episode(env, seed, rng):
    obs, mask = env.reset(seed=seed)
    rewards, steps, done = [], 0, False
    while not done:
        assert mask.any()
        obs, mask, reward, done = env.step(int(rng.choice(np.flatnonzero(mask))))
        rewards.append(reward)
        steps += 1
        assert steps < 20000
    return rewards


def test_reset_returns_obs_and_mask():
    obs, mask = CantStopEnv().reset(seed=0)
    assert obs.shape == (OBS_SIZE,) and mask.shape == (NUM_ACTIONS,) and mask.any()


def test_reward_is_sparse_and_positive_only_on_the_winning_step():
    env = CantStopEnv()
    rng = np.random.default_rng(0)
    for seed in range(20):
        rewards = random_episode(env, seed, rng)
        assert rewards[-1] == 1.0
        assert all(r == 0.0 for r in rewards[:-1])
        assert env.winner is not None


def test_same_seed_and_actions_give_same_game():
    def run():
        env = CantStopEnv()
        rng = np.random.default_rng(5)
        return random_episode(env, 9, rng), env.winner
    assert run() == run()


def test_illegal_action_and_step_after_done_raise():
    env = CantStopEnv()
    obs, mask = env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(int(np.flatnonzero(~mask)[0]))
    random_episode(env, 2, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        env.step(0)
