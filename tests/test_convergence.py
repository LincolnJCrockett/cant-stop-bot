import math

import torch

from cantstop.convergence import collect_probe_states, kl_divergence
from cantstop.env import NUM_ACTIONS
from cantstop.reinforce import train


def logp_over(legal_probs: list[float], legal_idx: list[int]):
    """A (1, 79) log-probability row and mask from probabilities on legal actions."""
    logp = torch.full((1, NUM_ACTIONS), float("-inf"))
    mask = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    for p, i in zip(legal_probs, legal_idx):
        logp[0, i] = math.log(p)
        mask[0, i] = True
    return logp, mask


def test_kl_of_identical_policies_is_zero():
    logp, mask = logp_over([0.5, 0.3, 0.2], [1, 4, 9])
    assert kl_divergence(logp, logp, mask) == 0.0


def test_kl_known_value_and_no_nan_from_masked_actions():
    old, mask = logp_over([0.5, 0.5], [2, 6])
    new, _ = logp_over([0.9, 0.1], [2, 6])
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)   # 0.5108
    assert abs(kl_divergence(old, new, mask) - expected) < 1e-5


def test_kl_is_nonnegative_and_not_symmetric():
    a, mask = logp_over([0.7, 0.2, 0.1], [0, 1, 2])
    b, _ = logp_over([0.2, 0.3, 0.5], [0, 1, 2])
    assert kl_divergence(a, b, mask) > 0
    assert kl_divergence(a, b, mask) != kl_divergence(b, a, mask)


def test_probe_states_are_fixed_and_legal():
    obs1, masks1 = collect_probe_states(40, seed=1)
    obs2, masks2 = collect_probe_states(40, seed=1)
    assert obs1.shape == (40, 68) and masks1.shape == (40, NUM_ACTIONS)
    assert torch.equal(obs1, obs2) and torch.equal(masks1, masks2)
    assert masks1.any(dim=1).all()                    # every state has a legal move


def test_training_stops_early_once_behaviour_is_stable(capsys):
    # A huge threshold counts every evaluation as "calm", so training should
    # stop after `patience` evaluations instead of running all 50 iterations.
    train(iters=50, games_per_iter=1, eval_every=1, eval_games=2, seed=0,
          stop_kl=1e9, patience=2)
    out = capsys.readouterr().out
    assert out.count("iter ") == 2 and "Converged" in out
