"""Baseline bots. A bot is a policy: (state, rng) -> Action.

The first two baselines pick pairings at random and differ only in their
stop/continue rule; the heuristic bot also chooses pairings with a score. Their win rates are the yardstick the trained agent
has to beat.
"""

from cantstop.actions import CONTINUE, STOP, Action, legal_actions
from cantstop.state import COLUMN_LENGTHS, State, col_index


def column_progress_this_turn(state: State) -> float:
    """Progress gained this turn, counted in fractions of a column.

    A runner's gain is (position - banked progress) / column length, so
    3 steps on column 2 (length 3) count as 1.0 -- a whole column --
    while 3 steps on column 7 (length 13) count as about 0.23.
    """
    banked = state.progress[state.current_player]
    total = 0.0
    for column, position in state.runners:
        gained = position - banked[col_index(column)]
        total += gained / COLUMN_LENGTHS[column]
    return total


def make_stop_after_column_progress(threshold: float):
    """Stop once this turn's column-fraction progress reaches `threshold`.

    Unlike a raw step count, this treats a step on a short column (hard to
    roll, big payoff) as worth more than a step on a long one.
    """
    def policy(state: State, rng):
        actions = legal_actions(state)
        if state.dice is not None:                  # choosing a pairing
            return actions[rng.integers(len(actions))]
        if column_progress_this_turn(state) >= threshold:
            return STOP
        return CONTINUE
    return policy


def steps_gained_this_turn(state: State) -> int:
    """Raw steps gained this turn, summed over runners.

    Runner positions are absolute, so subtracting the player's banked
    progress on that column leaves only this turn's steps.
    """
    banked = state.progress[state.current_player]
    return sum(position - banked[col_index(column)]
               for column, position in state.runners)


def make_stop_after_n(n: int):
    """Stop once this turn has gained `n` raw steps, whatever the columns.

    Every step counts the same, so unlike the column-progress bot it does
    not care that a step on column 2 is worth more than one on column 7.
    """
    def policy(state: State, rng):
        actions = legal_actions(state)
        if state.dice is not None:                  # choosing a pairing
            return actions[rng.integers(len(actions))]
        return STOP if steps_gained_this_turn(state) >= n else CONTINUE
    return policy


# --- Heuristic bot ---------------------------------------------------------
# Weights for scoring a PLAY. They are guesses; tune them by tournament.
W_EASE = 1.0       # per step, scaled by how easy the column is to roll
W_EXISTING = 2.0   # column already has a runner (saves a scarce runner)
W_CAPTURE = 5.0    # the play brings a runner to the top of its column
_LONGEST = max(COLUMN_LENGTHS.values())    # 13, the length of column 7


def play_score(state: State, action: Action) -> float:
    """How attractive a PLAY action is. Higher is better.

    - Ease: column length is a proxy for how likely the sum is (7 is the
      longest column because it is the most common sum), so steps on common
      columns score higher. Playing two sums scores higher than one.
    - Existing runner: using a runner already placed does not consume one of
      the 3 runners and keeps the bust risk down.
    - Capture: finishing a column is what wins the game.
    """
    banked = state.progress[state.current_player]
    score = 0.0
    for column in action.columns:
        score += W_EASE * COLUMN_LENGTHS[column] / _LONGEST
    for column in set(action.columns):
        start = state.runner_position(column)
        if start is None:
            start = banked[col_index(column)]
        else:
            score += W_EXISTING
        steps = action.columns.count(column)
        if start + steps >= COLUMN_LENGTHS[column]:
            score += W_CAPTURE
    return score


def choose_play(state: State, rng):
    """Highest-scoring legal PLAY; ties are broken at random."""
    actions = legal_actions(state)
    scores = [play_score(state, a) for a in actions]
    best = max(scores)
    top = [a for a, s in zip(actions, scores) if s == best]
    return top[rng.integers(len(top))]


def make_heuristic_bot(n: int):
    """Score-based pairing choice + the stop-after-n-steps rule."""
    def policy(state: State, rng):
        if state.dice is not None:
            return choose_play(state, rng)
        return STOP if steps_gained_this_turn(state) >= n else CONTINUE
    return policy
