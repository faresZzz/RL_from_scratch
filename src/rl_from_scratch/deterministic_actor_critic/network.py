"""Neural networks for DDPG and TD3: deterministic actor and Q(s,a) critic.

Deterministic actor: μ_θ(s) → a ∈ [action_low, action_high] via rescaled tanh.
Continuous critic: Q_φ(s, a) → scalar, input cat([s, a]).
Twin Q-Network: two independent critics for TD3 (mitigates overestimation).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """Initialize the weights of a linear layer with the orthogonal method."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class DeterministicActor(nn.Module):
    """Deterministic policy μ_θ(s) for bounded continuous action spaces.

    Produces an action via an MLP followed by tanh, rescaled from the bounds
    [-1, 1] to [action_low, action_high] using registered buffers.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    hidden_dim:
        Width of each hidden layer (256 by default for MuJoCo).
    action_low:
        Lower bound of the action space (scalar or vector).
    action_high:
        Upper bound of the action space (scalar or vector).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2)),
            nn.ReLU(),
        )
        self.action_head = _orthogonal_init(
            nn.Linear(hidden_dim, action_dim), gain=0.01
        )

        # Rescale the tanh output ∈ [-1, 1] to [action_low, action_high]
        action_scale = torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32
        )
        action_bias = torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32
        )
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the deterministic action μ_θ(obs).

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Actions of shape ``(batch, action_dim)`` in [action_low, action_high].
        """
        features = self.trunk(obs)
        raw = torch.tanh(self.action_head(features))
        return raw * self.action_scale + self.action_bias


class ContinuousQNetwork(nn.Module):
    """Q(s, a;φ) network for continuous action spaces.

    Concatenates the observation and action as input, produces a scalar Q value.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the action space.
    hidden_dim:
        Width of each hidden layer.
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
        """Evaluate Q(s, a) for a batch of transitions.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.
        action:
            Action tensor of shape ``(batch, action_dim)``.

        Returns
        -------
        torch.Tensor
            Q values of shape ``(batch,)``.
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Pair of independent Q networks for TD3 (overestimation mitigation).

    Uses two identical critics but with different weights. The target
    is built with the minimum of the two to reduce the positive bias.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the action space.
    hidden_dim:
        Width of each hidden layer.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = ContinuousQNetwork(obs_dim, action_dim, hidden_dim)
        self.q2 = ContinuousQNetwork(obs_dim, action_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q1(s, a) and Q2(s, a) simultaneously.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.
        action:
            Action tensor of shape ``(batch, action_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(q1, q2)`` each of shape ``(batch,)``.
        """
        return self.q1(obs, action), self.q2(obs, action)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate only Q1(s, a) — used for the TD3 actor loss.

        Avoids the unnecessary computation of Q2 during the actor update.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``(batch, obs_dim)``.
        action:
            Action tensor of shape ``(batch, action_dim)``.

        Returns
        -------
        torch.Tensor
            Q1 values of shape ``(batch,)``.
        """
        return self.q1(obs, action)
