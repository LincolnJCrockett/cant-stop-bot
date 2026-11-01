from cantstop.actions import legal_actions
from cantstop.play import describe, make_human_policy, play_game, random_policy, render
from cantstop.state import new_game


def test_random_vs_random_completes_with_winner():
    out = []
    final = play_game([random_policy, random_policy], seed=3, print_fn=out.append)
    assert final.is_over
    assert any("wins!" in line for line in out)


def test_human_policy_reprompts_on_bad_input_then_accepts():
    replies = iter(["x", "99", "1"])
    out = []
    human = make_human_policy(input_fn=lambda _: next(replies), print_fn=out.append)
    # Build a state with a pending roll so there are plays to choose from.
    from dataclasses import replace
    s = replace(new_game(), dice=(1, 2, 3, 4))
    action = human(s, None)
    assert action == legal_actions(s)[0]
    assert sum("Not a valid choice" in line for line in out) == 2


def test_human_can_play_whole_game_with_scripted_input():
    # Always pick option 1 (first legal action) against a seeded random bot.
    out = []
    human = make_human_policy(input_fn=lambda _: "1", print_fn=out.append)
    final = play_game([human, random_policy], seed=5, print_fn=out.append)
    assert final.is_over


def test_render_and_describe_smoke():
    text = render(new_game())
    assert "Col" in text and "12" in text
    from cantstop.actions import STOP, CONTINUE, Action, Kind
    assert "Stop" in describe(STOP) and "Continue" in describe(CONTINUE)
    assert describe(Action(Kind.PLAY, (6, 6))) == "Advance column 6 twice"
    assert describe(Action(Kind.PLAY, (6, 8))) == "Advance column 6 and 8"
