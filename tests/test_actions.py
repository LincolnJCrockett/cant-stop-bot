from dataclasses import replace

from cantstop.actions import CONTINUE, STOP, Action, Kind, legal_actions
from cantstop.state import new_game


def play(*cols):
    return Action(Kind.PLAY, tuple(cols))


def with_dice(dice, **changes):
    return replace(new_game(), dice=dice, **changes)


def owned(*columns, player=1):
    """owners tuple with the given columns captured by `player`."""
    owners = [None] * 11
    for c in columns:
        owners[c - 2] = player
    return tuple(owners)


def test_new_game_has_no_choice_and_no_winner():
    s = new_game()
    assert legal_actions(s) == []
    assert s.winner is None and not s.is_over


def test_fresh_roll_offers_every_distinct_pairing():
    # 1,2,3,4 -> (3,7), (4,6), (5,5)
    assert legal_actions(with_dice((1, 2, 3, 4))) == [
        play(3, 7), play(4, 6), play(5, 5)]


def test_identical_pairings_collapse_to_one_action():
    # 1,1,1,1 -> (2,2) three times
    assert legal_actions(with_dice((1, 1, 1, 1))) == [play(2, 2)]


def test_can_choose_one_sum_pairing_over_two_sum_pairing():
    # Runners on 3 and 7, one free. Roll 1,2,3,4: (3,7) both playable,
    # (4,6) both need a new runner but only one is free -> single sums,
    # (5,5) needs one new runner -> playable double.
    s = with_dice((1, 2, 3, 4), runners=((3, 0), (7, 0)))
    acts = legal_actions(s)
    assert play(3, 7) in acts
    assert play(4) in acts and play(6) in acts


def test_both_playable_is_mandatory_when_runners_available():
    s = with_dice((1, 2, 3, 4))   # 3 free runners
    acts = legal_actions(s)
    assert play(3) not in acts and play(7) not in acts


def test_one_free_runner_two_new_columns_choose_one():
    s = with_dice((1, 2, 3, 4), runners=((2, 0), (12, 0)))
    acts = legal_actions(s)
    assert play(3) in acts and play(7) in acts
    assert play(3, 7) not in acts


def test_all_runners_placed_and_no_match_is_bust():
    s = with_dice((1, 2, 3, 4), runners=((2, 0), (11, 0), (12, 0)))
    assert legal_actions(s) == []


def test_captured_column_not_playable():
    # 3 and 7 captured -> (3,7) pairing dead; (4,6),(5,5) remain.
    s = with_dice((1, 2, 3, 4), owners=owned(3, 7))
    assert legal_actions(s) == [play(4, 6), play(5, 5)]


def test_own_captured_column_also_closed():
    s = with_dice((1, 2, 3, 4), owners=owned(3, 7, player=0))
    assert play(3, 7) not in legal_actions(s)


def test_double_sum_one_step_from_top_still_playable():
    # Column 2 has length 3; runner at 2. (2,2) is legal (advances 1, capped).
    s = with_dice((1, 1, 1, 1), runners=((2, 2),))
    assert legal_actions(s) == [play(2, 2)]


def test_runner_at_top_not_playable_again():
    s = with_dice((1, 1, 1, 1), runners=((2, 3),))
    assert legal_actions(s) == []


def test_runner_starting_from_nonzero_progress_is_playable():
    s = with_dice((1, 1, 1, 1), runners=((2, 1),))
    assert legal_actions(s) == [play(2, 2)]


def test_runner_on_column_does_not_need_free_runner():
    # All 3 runners placed; roll hits one of them (5,5 on column 5).
    s = with_dice((2, 3, 2, 3), runners=((5, 0), (6, 0), (8, 0)))
    assert play(5, 5) in legal_actions(s)


def test_stop_forbidden_before_first_successful_roll():
    s = new_game()
    assert STOP not in legal_actions(s)


def test_after_successful_roll_stop_or_continue():
    s = replace(new_game(), rolled=True, runners=((6, 1),))
    assert legal_actions(s) == [CONTINUE, STOP]


def test_game_over_has_no_actions():
    s = replace(new_game(), owners=owned(2, 3, 4, player=1), rolled=True)
    assert s.winner == 1 and s.is_over
    assert legal_actions(s) == []
