"""Neural networks for SAC: squashed Gaussian actor and Q(s,a) critics.

Stochastic actor: π_θ(a|s) — Gaussian distribution with reparameterization
trick and tanh squashing, producing actions in [action_low, action_high].
Continuous critic: Q_φ(s, a) → scalar, local to the SAC chapter.
Twin Q-Network: two independent critics to reduce SAC overestimation.
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
    """Initialize the weights of a linear layer with the orthogonal method."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ContinuousQNetwork(nn.Module):
    """Critic Q(s, a;phi) for continuous action spaces.

    Concatenates the observation and action as input, then produces a scalar
    value via a small MLP. The local duplication keeps SAC self-contained.
    """

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
        """Evaluate only Q1(s, a)."""
        return self.q1(obs, action)


class SquashedGaussianActor(nn.Module):
    """Squashed Gaussian stochastic policy π_θ(a|s) for SAC.

    Parameterizes a normal distribution N(μ_θ(s), σ_θ(s)) in the
    unsquashed space, then applies tanh to bound the actions in [-1, 1]
    before rescaling to [action_low, action_high].

    The reparameterization trick allows propagating gradients through
    the stochastic sampling. The log-probability correction for the
    tanh squashing is applied exactly.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    hidden_dim:
        Width of each hidden layer (256 by default for MuJoCo).
    action_low:
        Lower bound of the action space. Can be a scalar or a
        vector of shape ``(action_dim,)``.
    action_high:
        Upper bound of the action space. Can be a scalar or a
        vector of shape ``(action_dim,)``.
    log_std_min:
        Minimum value of the log standard deviation (clamp).
    log_std_max:
        Maximum value of the log standard deviation (clamp).
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

        # MLP trunk shared between mean_head and log_std_head
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
        )

        # Separate heads for mean and log standard deviation
        self.mean_head = _orthogonal_init(
            nn.Linear(hidden_dim, action_dim), gain=0.01
        )
        self.log_std_head = _orthogonal_init(
            nn.Linear(hidden_dim, action_dim), gain=0.01
        )

        # Rescaling the tanh output ∈ [-1, 1] to [action_low, action_high]
        action_low_t = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32)
        action_scale = (action_high_t - action_low_t) / 2.0
        action_bias = (action_high_t + action_low_t) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and log standard deviation of the policy.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mean, log_std)`` each of shape ``(batch, action_dim)``.
            ``log_std`` is clamped to [log_std_min, log_std_max].
        """
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action via the reparameterization trick + tanh squashing.

        Steps:
        1. Compute μ and log σ from the network.
        2. Reparameterization: u = μ + σ * ε,  ε ~ N(0, I).
        3. Squashing: a_raw = tanh(u).
        4. Log-prob correction: log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u) + ε).
        5. Rescaling: a = a_raw * scale + bias.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ``action``: actions sampled in [action_low, action_high],
              shape ``(batch, action_dim)``.
            - ``log_prob``: corrected log-probabilities, shape ``(batch,)``.
            - ``deterministic_action``: deterministic action tanh(μ) rescaled,
              shape ``(batch, action_dim)``.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterization trick: u = μ + σ * ε
        normal = dist.Normal(mean, std)
        u = normal.rsample()

        # Tanh squashing
        action_raw = torch.tanh(u)

        # Gaussian log-probability - tanh squashing correction
        gaussian_log_prob = normal.log_prob(u)
        # Jacobian correction: log|det(∂a/∂u)| = Σ log(1 - tanh²(u) + ε)
        log_prob = (
            gaussian_log_prob
            - torch.log(1.0 - action_raw.pow(2) + 1e-6)
        ).sum(dim=-1)

        # Rescaling to [action_low, action_high]
        action = action_raw * self.action_scale + self.action_bias

        # Deterministic action (mode) for evaluation
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, deterministic_action
