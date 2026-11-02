"""Collecting self-play games for training."""

from dataclasses import dataclass, field

import numpy as np
import torch

from cantstop.env import CantStopEnv
from cantstop.model import PolicyNet, sample_action_index


@dataclass
class Episode:
    """One finished game, as parallel lists with one entry per decision."""
    observations: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    actions: list = field(default_factory=list)    # chosen action indices
    players: list = field(default_factory=list)    # who made each decision
    winner: int | None = None

    def __len__(self) -> int:
        return len(self.actions)


def play_episode(net: PolicyNet, env: CantStopEnv, seed: int) -> Episode:
    """Self-play one game: the same network chooses for both seats, sampling
    from its own probabilities (that sampling is how it explores)."""
    rng = np.random.default_rng(seed)
    episode = Episode()
    obs, mask = env.reset(seed=seed)
    done = False
    while not done:
        action = sample_action_index(net, obs, mask, rng)
        episode.observations.append(obs)
        episode.masks.append(mask)
        episode.actions.append(action)
        episode.players.append(env.current_player)
        obs, mask, _reward, done = env.step(action)
    episode.winner = env.winner
    return episode


def to_tensors(episodes: list[Episode], returns: list[np.ndarray]):
    """Stack many games into batch tensors for one gradient step."""
    obs = torch.as_tensor(np.concatenate([np.stack(e.observations) for e in episodes]))
    masks = torch.as_tensor(np.concatenate([np.stack(e.masks) for e in episodes]))
    actions = torch.as_tensor(np.concatenate([e.actions for e in episodes]),
                              dtype=torch.int64)
    g = torch.as_tensor(np.concatenate(returns), dtype=torch.float32)
    return obs, masks, actions, g
