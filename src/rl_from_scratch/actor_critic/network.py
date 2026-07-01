"""Neural networks for A2C: Gaussian policy and value critic."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """Initialize the weights of a linear layer with the orthogonal method."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class GaussianActor(nn.Module):
    """Gaussian policy π(a|s;θ) = N(μ_θ(s), σ²) for continuous actions.

    The mean μ is produced by a two-hidden-layer MLP. The standard deviation
    σ = exp(log_std) is a learned parameter independent of the state (a common
    strategy in A2C/PPO for continuous spaces).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    hidden_dim:
        Width of each hidden layer (256 by default for MuJoCo).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.Tanh(),
        )
        self.mean_head = _orthogonal_init(
            nn.Linear(hidden_dim, action_dim), gain=0.01
        )
        # log_std shared across all action dimensions (independent of the state)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and standard deviation of the policy.

        Parameters
        ----------
        x:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(mean, std)`` each of shape ``(batch, action_dim)``.
        """
        features = self.trunk(x)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """Build the Gaussian distribution N(μ_θ(obs), σ²).

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.distributions.Normal
            Gaussian distribution over the action space.
        """
        mean, std = self.forward(obs)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    """Value function V(s;φ): observation → scalar estimate.

    Architecture identical to the actor's trunk to make potential weight
    sharing easier in the future.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    hidden_dim:
        Width of each hidden layer.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate V(s) for a batch of observations.

        Parameters
        ----------
        x:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Values of shape ``(batch,)`` after removing the last dimension.
        """
        return self.net(x).squeeze(-1)
