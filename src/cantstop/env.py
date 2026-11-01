"""Training-facing environment pieces: the fixed action space and the mask.

A neural network has one output per possible action, so we need a single
global numbering of every action that could ever exist. Most are illegal at
any given moment; the boolean *mask* says which ones are legal right now.
"""

from itertools import combinations_with_replacement

import numpy as np

from cantstop.actions import CONTINUE, STOP, Action, Kind, legal_actions
from cantstop.engine import roll_until_decision
from cantstop.engine import step as engine_step
from cantstop.state import (COLUMN_LENGTHS, COLUMNS, NUM_RUNNERS, State,
                            col_index, new_game)

# Every action, in a fixed order. combinations_with_replacement over the
# sorted columns yields exactly the sorted tuples Action.columns uses:
#   (a, a)  - 11 double steps      (a, b), a < b  - 55 pairs
# plus 11 single-column plays and the 2 non-play actions.
_SINGLES = [(c,) for c in COLUMNS]
_PAIRS = list(combinations_with_replacement(COLUMNS, 2))
ALL_ACTIONS: tuple[Action, ...] = tuple(
    [Action(Kind.PLAY, cols) for cols in _SINGLES + _PAIRS] + [CONTINUE, STOP]
)
NUM_ACTIONS = len(ALL_ACTIONS)           # 11 + 66 + 2 = 79

# Action -> index. Works because Action is frozen (hashable).
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ALL_ACTIONS)}


def action_to_index(action: Action) -> int:
    return ACTION_INDEX[action]


def index_to_action(index: int) -> Action:
    return ALL_ACTIONS[index]


# --- Observation encoding -------------------------------------------------
# Layout (all float32, all in [0, 1]): six blocks of 11 (one value per
# column 2..12, in order), then 2 scalars.
#   0  my banked progress / column length
#   1  opponent's banked progress / column length
#   2  my runner's position / column length   (0 if no runner there)
#   3  runner flag: 1 if I have a runner on the column
#   4  captured by me
#   5  captured by the opponent
#   6  free runners / 3
#   7  phase: 1 = choosing a pairing (dice pending), 0 = stop/continue
OBS_SIZE = 6 * len(COLUMNS) + 2

_LENGTHS = np.array([COLUMN_LENGTHS[c] for c in COLUMNS], dtype=np.float32)


def encode_observation(state: State) -> np.ndarray:
    """Turn a State into a fixed-size vector, from the point of view of the
    player about to act. "Mine" and "opponent's" are decided by
    state.current_player, so one network can play either seat."""
    assert state.num_players == 2, "encoder supports 2 players"
    me = state.current_player
    opp = 1 - me

    my_progress = np.array(state.progress[me], dtype=np.float32) / _LENGTHS
    opp_progress = np.array(state.progress[opp], dtype=np.float32) / _LENGTHS

    runner_pos = np.zeros(len(COLUMNS), dtype=np.float32)
    runner_flag = np.zeros(len(COLUMNS), dtype=np.float32)
    for column, position in state.runners:
        runner_pos[col_index(column)] = position / COLUMN_LENGTHS[column]
        runner_flag[col_index(column)] = 1.0

    captured_me = np.array([o == me for o in state.owners], dtype=np.float32)
    captured_opp = np.array([o == opp for o in state.owners], dtype=np.float32)

    scalars = np.array(
        [state.free_runners / NUM_RUNNERS, float(state.dice is not None)],
        dtype=np.float32,
    )
    return np.concatenate([my_progress, opp_progress, runner_pos, runner_flag,
                           captured_me, captured_opp, scalars])


def legal_mask(state: State) -> np.ndarray:
    """Boolean array of length NUM_ACTIONS: True where the action is legal."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal_actions(state):
        mask[ACTION_INDEX[action]] = True
    return mask


class CantStopEnv:
    """The interface a training loop uses: reset() and step(action_index).

    Self-play: the environment hands control to whichever player must decide
    next, so every observation is from `current_player`'s point of view and
    the same network can play both seats.

    Reward is sparse. step() returns +1 when the move just made wins the game
    (only the acting player can win) and 0 otherwise. The loser's -1 is not
    given here: the training code assigns it from `winner` when it builds
    each player's returns.
    """

    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.state: State | None = None
        self.rng: np.random.Generator | None = None

    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def winner(self) -> int | None:
        return self.state.winner

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Start a new game. Returns (observation, legal mask)."""
        self.rng = np.random.default_rng(seed)
        self.state = roll_until_decision(new_game(self.num_players), self.rng)
        return encode_observation(self.state), legal_mask(self.state)

    def step(self, action_index: int):
        """Apply the chosen action. Returns (obs, mask, reward, done)."""
        if self.state is None or self.state.is_over:
            raise RuntimeError("call reset() before step()")
        actor = self.state.current_player
        self.state = engine_step(self.state, index_to_action(action_index), self.rng)
        done = self.state.is_over
        reward = 1.0 if done and self.state.winner == actor else 0.0
        return encode_observation(self.state), legal_mask(self.state), reward, done
