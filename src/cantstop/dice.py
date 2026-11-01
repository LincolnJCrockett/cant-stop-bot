"""Dice rolling and pairing.

Pure functions only: randomness comes from a caller-supplied
``numpy.random.Generator`` so every game is reproducible from a seed.
"""

import numpy as np

# The 3 ways to split 4 dice (by index) into two pairs.
PAIRING_INDICES = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def roll_dice(rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Roll 4 six-sided dice."""
    a, b, c, d = rng.integers(1, 7, size=4)
    return int(a), int(b), int(c), int(d)


def pairings(dice: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Return the 3 pairings of the dice as (sum, sum) tuples.

    Order within a tuple is sorted so (7, 5) and (5, 7) look the same;
    duplicates across pairings are kept (they are distinct choices of dice
    even if they give the same sums), and the list always has length 3.
    """
    return [
        tuple(sorted((dice[i] + dice[j], dice[k] + dice[m])))
        for (i, j), (k, m) in PAIRING_INDICES
    ]
