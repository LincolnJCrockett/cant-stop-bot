"""The policy network and the pieces that turn its output into actions.

Torch is used for tensors, autograd and nn.Linear only; the masking,
sampling and (in reinforce.py) the loss are written by hand.
"""

import numpy as np
import torch
from torch import nn

from cantstop.env import NUM_ACTIONS, OBS_SIZE, index_to_action, legal_mask
from cantstop.env import encode_observation
from cantstop.state import State


class PolicyNet(nn.Module):
    """observation (68 floats) -> one raw score ("logit") per action (79).

    A plain multi-layer perceptron: two hidden layers with ReLU between.
    The scores are NOT probabilities yet; masking and softmax come next.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log pi(action | state) for every action, with illegal actions removed.

    Setting an illegal action's logit to -inf makes its softmax probability
    exactly 0, and the legal probabilities sum to 1. log_softmax is the
    numerically stable way to compute log(softmax(x)).
    """
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def sample_action_index(net: PolicyNet, obs: np.ndarray, mask: np.ndarray,
                        rng: np.random.Generator, greedy: bool = False) -> int:
    """Pick an action index for one decision (no gradients needed here)."""
    with torch.no_grad():
        logits = net(torch.as_tensor(obs).unsqueeze(0))
        log_probs = masked_log_probs(logits, torch.as_tensor(mask).unsqueeze(0))
    probs = log_probs.exp().squeeze(0).numpy().astype(np.float64)
    if greedy:
        return int(probs.argmax())
    return int(rng.choice(NUM_ACTIONS, p=probs / probs.sum()))


def make_policy(net: PolicyNet, greedy: bool = False):
    """Wrap a network as a policy (state, rng) -> Action, so it can play in
    the tournament harness next to the hand-written bots."""
    def policy(state: State, rng: np.random.Generator):
        index = sample_action_index(net, encode_observation(state),
                                    legal_mask(state), rng, greedy)
        return index_to_action(index)
    return policy
