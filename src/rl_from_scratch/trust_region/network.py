"""Local network definitions for trust-region algorithms."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActor(nn.Module):
    """Gaussian policy with state-independent log standard deviation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
        )
        self.mean_head = _orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    """Scalar value network V(s)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
