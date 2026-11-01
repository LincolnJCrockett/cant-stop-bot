"""The engine: the functions that move a game from one State to the next.

A game alternates between two kinds of events:
  - chance:   the dice are rolled         -> roll()   (needs an RNG)
  - decision: a player picks an Action    -> apply()  (fully deterministic)

Every function returns a NEW State; the old one is never modified.
"""

from dataclasses import replace

import numpy as np

from cantstop.actions import Action, Kind, legal_actions, play_actions
from cantstop.dice import roll_dice
from cantstop.state import COLUMN_LENGTHS, State, col_index


def needs_roll(state: State) -> bool:
    """True when the next event is a dice roll rather than a decision."""
    return not state.is_over and state.dice is None and not state.rolled


def _end_turn(state: State) -> State:
    """Discard runners and pass the turn to the next player."""
    return replace(
        state,
        runners=(),
        dice=None,
        rolled=False,
        current_player=(state.current_player + 1) % state.num_players,
    )


def roll(state: State, rng: np.random.Generator) -> State:
    """Roll the dice. If nothing is playable the player busts (Rule 5):
    runner progress is lost and the turn passes. Otherwise the dice are
    stored in the state, waiting for the player to choose a pairing."""
    assert needs_roll(state), "roll() called when a decision is pending"
    rolled_state = replace(state, dice=roll_dice(rng))
    if not play_actions(rolled_state):
        return _end_turn(state)          # bust: the roll is discarded
    return rolled_state


def _play(state: State, action: Action) -> State:
    """Advance one step on each column in the action (Rule 4)."""
    player = state.current_player
    positions = dict(state.runners)
    for column in action.columns:
        # A new runner starts from the player's permanent progress.
        current = positions.get(column, state.progress[player][col_index(column)])
        positions[column] = min(current + 1, COLUMN_LENGTHS[column])
    return replace(
        state,
        runners=tuple(sorted(positions.items())),
        dice=None,
        rolled=True,          # now facing a stop/continue decision
    )


def _stop(state: State) -> State:
    """Bank runner progress, capture finished columns, pass the turn (Rule 6/7)."""
    player = state.current_player
    progress = list(state.progress[player])
    owners = list(state.owners)
    for column, position in state.runners:
        progress[col_index(column)] = position
        if position == COLUMN_LENGTHS[column]:
            owners[col_index(column)] = player
    banked = replace(
        state,
        progress=state.progress[:player] + (tuple(progress),) + state.progress[player + 1:],
        owners=tuple(owners),
    )
    if banked.is_over:       # winner stays as the current player; no next turn
        return replace(banked, runners=(), dice=None, rolled=False)
    return _end_turn(banked)


def apply(state: State, action: Action) -> State:
    """Apply a player's decision. Deterministic; raises on an illegal action."""
    if action not in legal_actions(state):
        raise ValueError(f"illegal action {action} in this state")
    if action.kind is Kind.PLAY:
        return _play(state, action)
    if action.kind is Kind.STOP:
        return _stop(state)
    # CONTINUE: the decision is made; a roll is now due.
    return replace(state, rolled=False)


def roll_until_decision(state: State, rng: np.random.Generator) -> State:
    """Roll (repeatedly, across busts and turn changes) until a player has a
    real decision to make or the game is over."""
    while needs_roll(state):
        state = roll(state, rng)
    return state


def step(state: State, action: Action, rng: np.random.Generator) -> State:
    """One full environment step: apply the decision, then run chance events
    until the next decision. This is what the M2 environment will wrap."""
    return roll_until_decision(apply(state, action), rng)
