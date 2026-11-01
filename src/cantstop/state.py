"""Game state: the board constants and an immutable snapshot of a game.

The board never changes, so it lives in constants. Everything that changes
during a game lives in ``State``. ``State`` is frozen: applying an action
produces a *new* State and never edits the old one. That keeps the engine
pure, makes tests easy, and means a game history is just a list of States.
"""

from dataclasses import dataclass

# --- Board constants (identical in every game) ---------------------------
# Column number -> number of steps to reach the top.
COLUMN_LENGTHS = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13,
                  8: 11, 9: 9, 10: 7, 11: 5, 12: 3}
COLUMNS = tuple(COLUMN_LENGTHS)          # (2, 3, ..., 12)
NUM_RUNNERS = 3
COLUMNS_TO_WIN = 3


def col_index(column: int) -> int:
    """Column numbers run 2..12; tuples are indexed 0..10."""
    return column - 2


# --- Game state (changes every move) ------------------------------------
@dataclass(frozen=True)
class State:
    # progress[player][col_index(column)] = that player's PERMANENT steps.
    progress: tuple[tuple[int, ...], ...]
    # owners[col_index(column)] = player who captured it, or None if open.
    owners: tuple[int | None, ...]
    current_player: int
    # This turn's provisional runners: sorted ((column, position), ...).
    # Position is absolute (it starts from the player's permanent progress).
    # Lost on a bust, copied into `progress` on a stop.
    runners: tuple[tuple[int, int], ...] = ()
    # The 4 dice if a roll is waiting for the player to choose a pairing.
    dice: tuple[int, int, int, int] | None = None
    # True while the player is facing a stop/continue decision, i.e. right
    # after a successful roll has been played. This is what makes stop
    # illegal before the first successful roll. False means "a roll is due".
    rolled: bool = False

    @property
    def num_players(self) -> int:
        return len(self.progress)

    @property
    def winner(self) -> int | None:
        """The player with 3+ captured columns, or None. Computed, not
        stored, so it can never disagree with `owners`."""
        for player in range(self.num_players):
            if self.owners.count(player) >= COLUMNS_TO_WIN:
                return player
        return None

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    @property
    def free_runners(self) -> int:
        return NUM_RUNNERS - len(self.runners)

    def runner_position(self, column: int) -> int | None:
        """Position of this turn's runner on `column`, or None."""
        for col, pos in self.runners:
            if col == column:
                return pos
        return None


def new_game(num_players: int = 2) -> State:
    """A fresh game: nobody has progress, player 0 to move."""
    return State(
        progress=tuple((0,) * len(COLUMNS) for _ in range(num_players)),
        owners=(None,) * len(COLUMNS),
        current_player=0,
    )
