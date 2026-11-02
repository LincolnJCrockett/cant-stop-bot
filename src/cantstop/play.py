"""Play Can't Stop in the terminal:  python -m cantstop.play

A *policy* is any function  (state, rng) -> Action.  A human typing at the
keyboard, a random bot, and (later) a trained network are all policies, so
the game loop below never needs to know who is playing.
"""

import argparse
from typing import Callable

import numpy as np

from cantstop.actions import Action, Kind, legal_actions
from cantstop.engine import apply, needs_roll, roll
from cantstop.state import COLUMN_LENGTHS, State, new_game

Policy = Callable[[State, np.random.Generator], Action]


# --- Display -------------------------------------------------------------
def render(state: State) -> str:
    """The board: per column, each player's banked steps, this turn's runner,
    and who (if anyone) has captured it."""
    me = state.current_player
    header = "Col  Len  " + "  ".join(f"P{p}" for p in range(state.num_players))
    lines = [header + "   Runner  Status"]
    for column, length in COLUMN_LENGTHS.items():
        banked = "  ".join(f"{state.progress[p][column - 2]:>2}" for p in range(state.num_players))
        pos = state.runner_position(column)
        runner = f"{pos:>2}" if pos is not None else " ."
        owner = state.owners[column - 2]
        status = f"captured by P{owner}" if owner is not None else ""
        lines.append(f"{column:>3}  {length:>3}  {banked}     {runner}    {status}")
    lines.append(f"Player {me}: {state.free_runners} runner(s) free")
    return "\n".join(lines)


def describe(action: Action) -> str:
    if action.kind is Kind.STOP:
        return "Stop (bank your progress)"
    if action.kind is Kind.CONTINUE:
        return "Continue (roll again)"
    cols = action.columns
    if len(cols) == 2 and cols[0] == cols[1]:
        return f"Advance column {cols[0]} twice"
    return "Advance column " + " and ".join(str(c) for c in cols)


# --- Policies ------------------------------------------------------------
def random_policy(state: State, rng: np.random.Generator) -> Action:
    actions = legal_actions(state)
    return actions[rng.integers(len(actions))]


def make_human_policy(input_fn=input, print_fn=print) -> Policy:
    def human(state: State, rng: np.random.Generator) -> Action:
        actions = legal_actions(state)
        print_fn("\n" + render(state))
        if state.dice is not None:
            print_fn(f"You rolled {state.dice}")
        for i, action in enumerate(actions, 1):
            print_fn(f"  {i}) {describe(action)}")
        while True:
            choice = input_fn("Choose (number, or q to quit): ").strip().lower()
            if choice == "q":
                raise KeyboardInterrupt
            if choice.isdigit() and 1 <= int(choice) <= len(actions):
                return actions[int(choice) - 1]
            print_fn("Not a valid choice.")
    human.is_human = True   # lets the game loop skip narrating human moves
    return human


def load_model_policy(path: str, greedy: bool = False) -> Policy:
    """Load a saved network checkpoint and wrap it as a policy."""
    import torch                       # imported here: torch is optional
    from cantstop.model import PolicyNet, make_policy
    net = PolicyNet()
    net.load_state_dict(torch.load(path))
    net.eval()
    return make_policy(net, greedy)


# --- Game loop -----------------------------------------------------------
def play_game(policies: list[Policy], seed: int | None = None,
              print_fn=print, verbose_bots: bool = True) -> State:
    """Run one game. policies[i] controls player i. Returns the final state."""
    rng = np.random.default_rng(seed)
    state = new_game(len(policies))
    while True:
        # Chance events: roll until someone must decide (announce busts).
        while needs_roll(state):
            before = state
            state = roll(state, rng)
            if state.dice is None:
                print_fn(f"Player {before.current_player} rolled and BUSTED "
                         f"- turn progress lost.")
        if state.is_over:
            print_fn("\n" + render(state))
            print_fn(f"\nPlayer {state.winner} wins!")
            return state
        player = state.current_player
        action = policies[player](state, rng)
        if verbose_bots and not getattr(policies[player], "is_human", False):
            print_fn(f"Player {player} (bot) rolled {state.dice}: {describe(action)}"
                     if state.dice is not None else
                     f"Player {player} (bot): {describe(action)}")
        state = apply(state, action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Can't Stop in the terminal.")
    parser.add_argument("--opponent", choices=["random", "human"], default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="path to a trained checkpoint to play against")
    parser.add_argument("--greedy", action="store_true",
                        help="model always picks its most likely move")
    args = parser.parse_args()
    human = make_human_policy()
    if args.model:
        opponent = load_model_policy(args.model, greedy=args.greedy)
    else:
        opponent = human if args.opponent == "human" else random_policy
    try:
        play_game([human, opponent], seed=args.seed)
    except KeyboardInterrupt:
        print("\nGame abandoned.")


if __name__ == "__main__":
    main()
