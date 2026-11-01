"""Actions and the legal-action generator."""

from dataclasses import dataclass
from enum import Enum

from cantstop.dice import pairings
from cantstop.state import COLUMN_LENGTHS, State


class Kind(Enum):
    PLAY = "play"          # apply a pairing's sums to the runners
    STOP = "stop"          # bank this turn's runner progress
    CONTINUE = "continue"  # roll again


@dataclass(frozen=True)
class Action:
    """One decision. Immutable and hashable, so actions can go in sets/dicts.

    For PLAY, `columns` are the columns to advance one step each, sorted:
      (6, 8)  two different columns
      (6, 6)  one column, two steps
      (6,)    a single sum (only one was playable, or the runner limit
              forced a choice)
    The dice themselves no longer matter once a pairing is chosen, and two
    pairings that give the same sums are the same action.
    """
    kind: Kind
    columns: tuple[int, ...] = ()


STOP = Action(Kind.STOP)
CONTINUE = Action(Kind.CONTINUE)


def _playable(state: State, column: int) -> bool:
    """Can a runner advance on `column` right now? (Rule 3)"""
    if state.owners[column - 2] is not None:       # captured by anyone
        return False
    pos = state.runner_position(column)
    if pos is None:                                 # needs a new runner
        return state.free_runners > 0
    return pos < COLUMN_LENGTHS[column]             # not already at the top


def play_actions(state: State) -> set[Action]:
    """Legal PLAY actions for the pending dice (Rule 4). Empty set = bust."""
    plays: set[Action] = set()
    for a, b in pairings(state.dice):
        pa, pb = _playable(state, a), _playable(state, b)
        if a == b:
            if pa:
                plays.add(Action(Kind.PLAY, (a, a)))
        elif pa and pb:
            both_new = state.runner_position(a) is None and state.runner_position(b) is None
            if both_new and state.free_runners == 1:
                # Only one runner left for two new columns: pick one.
                plays.add(Action(Kind.PLAY, (a,)))
                plays.add(Action(Kind.PLAY, (b,)))
            else:
                plays.add(Action(Kind.PLAY, (a, b)))   # both are mandatory
        elif pa:
            plays.add(Action(Kind.PLAY, (a,)))
        elif pb:
            plays.add(Action(Kind.PLAY, (b,)))
    return plays


def legal_actions(state: State) -> list[Action]:
    """Every action the current player may take in `state`.

    - game over            -> none
    - dice pending         -> the legal plays (none means the player busted)
    - rolled, no dice      -> STOP or CONTINUE
    - nothing rolled yet   -> none (no choice; the engine just rolls)
    """
    if state.is_over:
        return []
    if state.dice is not None:
        return sorted(play_actions(state), key=lambda x: x.columns)
    if state.rolled:
        return [CONTINUE, STOP]
    return []
