"""Neural network components for MBPO: squashed Gaussian actor and twin Q-networks.

Duplicated from rl_from_scratch.sac.network so that mbpo is fully self-contained
and obeys the cross-package isolation rule (REGLES §5).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributions as dist
import torch.nn as nn

__all__ = [
    "_orthogonal_init",
    "ContinuousQNetwork",
    "TwinQNetwork",
    "SquashedGaussianActor",
]


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """Initialise a linear layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ContinuousQNetwork(nn.Module):
    """Q(s, a; φ) critic for continuous action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(input_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate Q(s, a) for a batch of transitions."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Pair of independent Q-networks for the SAC twin-critic objective."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = ContinuousQNetwork(obs_dim, action_dim, hidden_dim)
        self.q2 = ContinuousQNetwork(obs_dim, action_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q1(s, a) and Q2(s, a) simultaneously."""
        return self.q1(obs, action), self.q2(obs, action)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate Q1(s, a) only."""
        return self.q1(obs, action)


class SquashedGaussianActor(nn.Module):
    """Stochastic squashed Gaussian policy π_θ(a|s) for SAC.

    Parameterises N(μ_θ(s), σ_θ(s)) in the pre-squash space, applies tanh
    to bound actions to [-1, 1], then rescales to [action_low, action_high].

    Parameters
    ----------
    obs_dim:
        Observation space dimensionality.
    action_dim:
        Action space dimensionality.
    hidden_dim:
        Hidden layer width.
    action_low:
        Lower bound of the action space (scalar or array).
    action_high:
        Upper bound of the action space (scalar or array).
    log_std_min:
        Minimum log-standard-deviation (clamp).
    log_std_max:
        Maximum log-standard-deviation (clamp).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared trunk
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
        )

        # Separate heads for mean and log-std
        self.mean_head = _orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.log_std_head = _orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)

        # Rescaling from tanh ∈ [-1, 1] to [action_low, action_high]
        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        action_scale = (action_high_t - action_low_t) / 2.0
        action_bias = (action_high_t + action_low_t) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy mean and log-std.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(mean, log_std)`` each of shape ``(batch, action_dim)``.
        """
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action via the reparameterization trick + tanh squashing.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            - ``action``              : sampled action in [action_low, action_high], shape (B, A).
            - ``log_prob``            : squash-corrected log-probability, shape (B,).
            - ``deterministic_action``: tanh(μ) rescaled, shape (B, A).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = dist.Normal(mean, std)
        u = normal.rsample()

        action_raw = torch.tanh(u)

        gaussian_log_prob = normal.log_prob(u)
        log_prob = (
            gaussian_log_prob - torch.log(1.0 - action_raw.pow(2) + 1e-6)
        ).sum(dim=-1)

        action = action_raw * self.action_scale + self.action_bias
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, deterministic_action
