"""Measuring whether a policy has stopped changing.

The training loss is not a progress signal, so we watch the policy's
*behaviour* instead: fix a set of probe positions, and compare the action
probabilities the network assigns there at two different moments.
"""

import numpy as np
import torch

from cantstop.bots import make_heuristic_colprog_bot
from cantstop.env import CantStopEnv, action_to_index
from cantstop.model import PolicyNet, masked_log_probs


def collect_probe_states(n: int = 500, seed: int = 12345, every: int = 7):
    """A fixed set of n realistic positions (observation + legal mask),
    taken from games between strong bots, one position out of every `every`
    decisions so they are spread across games and game phases."""
    bot = make_heuristic_colprog_bot(0.6)
    env = CantStopEnv()
    rng = np.random.default_rng(seed)
    obs_list, mask_list = [], []
    count, game = 0, 0
    while len(obs_list) < n:
        obs, mask = env.reset(seed=seed + game)
        game += 1
        done = False
        while not done and len(obs_list) < n:
            if count % every == 0:
                obs_list.append(obs)
                mask_list.append(mask)
            count += 1
            obs, mask, _r, done = env.step(action_to_index(bot(env.state, rng)))
    return (torch.as_tensor(np.stack(obs_list)),
            torch.as_tensor(np.stack(mask_list)))


def log_policy(net: PolicyNet, obs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """log pi(. | state) over all 79 actions for every probe state."""
    with torch.no_grad():
        return masked_log_probs(net(obs), masks)


def kl_divergence(logp_old: torch.Tensor, logp_new: torch.Tensor,
                  masks: torch.Tensor) -> float:
    """Mean over states of  sum_a P_old(a) * (log P_old(a) - log P_new(a)).

    Only legal actions are summed: illegal ones have probability 0 in both
    policies and would otherwise give 0 * (-inf) = nan.
    """
    p_old = logp_old.exp()
    terms = torch.where(masks, p_old * (logp_old - logp_new),
                        torch.zeros_like(p_old))
    return terms.sum(dim=1).mean().item()
