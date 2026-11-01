import numpy as np

from cantstop.dice import pairings, roll_dice


def test_roll_is_four_dice_in_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dice = roll_dice(rng)
        assert len(dice) == 4
        assert all(1 <= d <= 6 for d in dice)


def test_roll_is_deterministic_for_seed():
    assert roll_dice(np.random.default_rng(42)) == roll_dice(np.random.default_rng(42))


def test_pairings_example():
    # 1,2,3,4 -> (1+2,3+4)=(3,7); (1+3,2+4)=(4,6); (1+4,2+3)=(5,5)
    assert pairings((1, 2, 3, 4)) == [(3, 7), (4, 6), (5, 5)]


def test_pairings_sorted_and_total_preserved():
    rng = np.random.default_rng(1)
    for _ in range(200):
        dice = roll_dice(rng)
        ps = pairings(dice)
        assert len(ps) == 3
        for lo, hi in ps:
            assert lo <= hi
            assert lo + hi == sum(dice)
            assert 2 <= lo and hi <= 12
