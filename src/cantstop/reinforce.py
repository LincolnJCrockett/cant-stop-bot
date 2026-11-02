"""REINFORCE: the simplest policy-gradient algorithm, written by hand.

    python -m cantstop.reinforce --iters 100 --games 16 --seed 0

Loop: play a batch of self-play games -> give each decision the return
(+1 if its player won, -1 if lost) -> take one gradient step on
-log pi(action | state) * return -> repeat.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from cantstop.bots import make_heuristic_colprog_bot
from cantstop.convergence import collect_probe_states, kl_divergence, log_policy
from cantstop.env import CantStopEnv
from cantstop.model import PolicyNet, make_policy, masked_log_probs
from cantstop.play import random_policy
from cantstop.rollout import Episode, play_episode, to_tensors
from cantstop.tournament import play_match


def compute_returns(players: list[int], winner: int) -> np.ndarray:
    """Return for every decision of a finished game.

    The only reward is at the end (+1 win), so with no discounting every
    decision by the winner has return +1 and every decision by the loser -1.
    """
    return np.array([1.0 if p == winner else -1.0 for p in players],
                    dtype=np.float32)


def reinforce_loss(net: PolicyNet, obs: torch.Tensor, masks: torch.Tensor,
                   actions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """-mean( log pi(action | state) * return ) over every decision in the batch.

    Minimising this raises the probability of actions that came with a
    positive return and lowers it for negative returns. `returns` carries no
    gradient: it is a constant weight, only log pi depends on the network.
    """
    logits = net(obs)                                       # (B, 79) raw scores
    log_probs = masked_log_probs(logits, masks)             # illegal -> -inf
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)   # (B,)
    return -(chosen * returns).mean()


def evaluate(net: PolicyNet, games: int = 100, seed: int = 10_000) -> dict:
    """Win rate of the network (sampling from its policy) against baselines."""
    policy = make_policy(net)
    return {
        "random": play_match(policy, random_policy, games, seed).win_rate_a,
        "heur cp.6": play_match(policy, make_heuristic_colprog_bot(0.6),
                                games, seed).win_rate_a,
    }


def train(iters: int = 100, games_per_iter: int = 16, lr: float = 1e-3,
          seed: int = 0, eval_every: int = 10, eval_games: int = 100,
          save_path: str | None = None, stop_kl: float | None = None,
          patience: int = 3) -> PolicyNet:
    """Train. If stop_kl is set, stop early once the policy has stopped
    changing: at `patience` evaluations in a row, the KL divergence between
    the policy and its previous-evaluation self is below stop_kl."""
    torch.manual_seed(seed)                      # fixes the initial weights
    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    env = CantStopEnv()
    probe_obs, probe_masks = collect_probe_states()
    prev_logp = log_policy(net, probe_obs, probe_masks)   # policy at last eval
    calm_evals = 0
    start = time.time()
    for it in range(1, iters + 1):
        # 1. Play: collect a batch of self-play games.
        episodes = [play_episode(net, env, seed * 1_000_000 + it * 1000 + g)
                    for g in range(games_per_iter)]
        returns = [compute_returns(e.players, e.winner) for e in episodes]
        obs, masks, actions, g = to_tensors(episodes, returns)
        # 2. Learn: one gradient step on the whole batch.
        loss = reinforce_loss(net, obs, masks, actions, g)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % eval_every == 0 or it == iters:
            result = evaluate(net, eval_games)
            # How much did the policy's behaviour change since the last eval?
            logp = log_policy(net, probe_obs, probe_masks)
            kl = kl_divergence(prev_logp, logp, probe_masks)
            prev_logp = logp
            print(f"iter {it:4d}  vs random {result['random']:.0%}"
                  f"  vs heur cp.6 {result['heur cp.6']:.0%}"
                  f"  KL {kl:.4f}"
                  f"  ({time.time() - start:.0f}s)", flush=True)
            if stop_kl is not None:
                calm_evals = calm_evals + 1 if kl < stop_kl else 0
                if calm_evals >= patience:
                    print(f"Converged: behaviour stable for {patience} evaluations.")
                    break
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), save_path)
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with REINFORCE.")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--games", type=int, default=16, help="games per iteration")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--stop-kl", type=float, default=None,
                        help="stop early once KL between evals stays below this")
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    train(args.iters, args.games, args.lr, args.seed, args.eval_every,
          args.eval_games, save_path=args.save, stop_kl=args.stop_kl,
          patience=args.patience)


if __name__ == "__main__":
    main()
