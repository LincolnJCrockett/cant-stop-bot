"""Tournament harness: measure how strong a policy is, fairly and repeatably.

    python -m cantstop.tournament [--games 200] [--seed 0]

A policy is any function (state, rng) -> Action, so the trained agent will
plug in here exactly like the hand-written bots do.
"""

import argparse
from dataclasses import dataclass
from math import sqrt

import numpy as np

from cantstop.bots import (make_heuristic_bot, make_heuristic_colprog_bot,
                           make_stop_after_column_progress, make_stop_after_n)
from cantstop.engine import roll_until_decision, step
from cantstop.play import Policy, random_policy
from cantstop.state import new_game


def play_one_game(policies: list[Policy], seed: int) -> int:
    """Play a silent game and return the winning player's seat.

    One RNG, seeded from `seed`, drives both the dice and any random choices
    the bots make, so the whole game is reproducible from the seed.
    """
    rng = np.random.default_rng(seed)
    state = roll_until_decision(new_game(len(policies)), rng)
    while not state.is_over:
        action = policies[state.current_player](state, rng)
        state = step(state, action, rng)
    return state.winner


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (Wilson score interval).

    Meaning: if the bot's true win rate were outside this range, results
    like ours would be unlikely. It shrinks roughly as 1/sqrt(games), so
    four times as many games gives an interval half as wide.
    """
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class MatchResult:
    wins_a: int
    wins_b: int

    @property
    def games(self) -> int:
        return self.wins_a + self.wins_b

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games

    @property
    def interval_a(self) -> tuple[float, float]:
        return wilson_interval(self.wins_a, self.games)


def play_match(policy_a: Policy, policy_b: Policy, games: int = 200,
               seed: int = 0) -> MatchResult:
    """Play `games` games of A vs B. Game g uses seed `seed + g`, and the
    two policies swap seats every game so neither gets the first-move edge."""
    wins_a = 0
    for g in range(games):
        a_seat = g % 2
        policies = [policy_a, policy_b] if a_seat == 0 else [policy_b, policy_a]
        wins_a += play_one_game(policies, seed + g) == a_seat
    return MatchResult(wins_a, games - wins_a)


def round_robin(bots: dict[str, Policy], games: int = 200, seed: int = 0
                ) -> dict[tuple[str, str], MatchResult]:
    """Every bot against every other bot. results[(a, b)] is A's view of
    the A-vs-B match; the reverse pairing is filled in from the same games."""
    names = list(bots)
    results: dict[tuple[str, str], MatchResult] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = play_match(bots[a], bots[b], games, seed)
            results[(a, b)] = r
            results[(b, a)] = MatchResult(r.wins_b, r.wins_a)
    return results


def format_table(names: list[str], results) -> str:
    """Rows are the bot, columns its opponent; cells are the row bot's win
    rate. The last column is the row bot's average over all opponents."""
    width = max(len(n) for n in names) + 2
    lines = [" " * width + "".join(f"{n[:9]:>10}" for n in names) + "   average"]
    for a in names:
        cells, rates = [], []
        for b in names:
            if a == b:
                cells.append(f"{'-':>10}")
            else:
                rate = results[(a, b)].win_rate_a
                rates.append(rate)
                cells.append(f"{rate:>10.1%}")
        lines.append(f"{a:<{width}}" + "".join(cells) + f"   {np.mean(rates):.1%}")
    return "\n".join(lines)


def default_bots() -> dict[str, Policy]:
    return {
        "random": random_policy,
        "stopN=3": make_stop_after_n(3),
        "stopN=5": make_stop_after_n(5),
        "colprog.5": make_stop_after_column_progress(0.5),
        "heur N=5": make_heuristic_bot(5),
        "heur cp.6": make_heuristic_colprog_bot(0.6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin bot tournament.")
    parser.add_argument("--games", type=int, default=200, help="games per pairing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    bots = default_bots()
    results = round_robin(bots, args.games, args.seed)
    print(format_table(list(bots), results))
    lo, hi = wilson_interval(args.games // 2, args.games)
    print(f"\n{args.games} games per pairing: a 50% result is accurate to "
          f"about {lo:.0%}-{hi:.0%} (95% interval).")


if __name__ == "__main__":
    main()
