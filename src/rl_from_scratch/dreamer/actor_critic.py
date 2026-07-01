"""Actor-critic networks for DreamerV1, duplicated locally (no imports from sac).

Actor: squashed Gaussian that takes a latent *feature* vector (not a raw
observation) as input and outputs actions in [action_low, action_high].
Critic: MLP from feature vector to scalar state-value V(s).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Actor — Squashed Gaussian (reparameterised, tanh-bounded)
# ──────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Stochastic actor for continuous action spaces.

    Parameterises N(μ(feat), σ(feat)) in un-squashed space, then applies
    ``tanh`` and rescales to [action_low, action_high].

    Parameters
    ----------
    feat_dim:
        Dimensionality of the RSSM feature (deter_dim + stoch_dim).
    action_dim:
        Dimensionality of the action space.
    hidden_dim:
        Width of the MLP hidden layers.
    action_low:
        Lower bounds of the action space (array or scalar).
    action_high:
        Upper bounds of the action space (array or scalar).
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Scaling buffers: tanh ∈ [-1,1] → [action_low, action_high]
        low_t = torch.as_tensor(action_low, dtype=torch.float32)
        high_t = torch.as_tensor(action_high, dtype=torch.float32)
        scale = (high_t - low_t) / 2.0
        bias = (high_t + low_t) / 2.0
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(
        self, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean_raw, log_std) before squashing."""
        h = self.trunk(feat)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action via the reparameterisation trick.

        1. Compute μ, log σ.
        2. Reparameterise: u = μ + σ * ε, ε ~ N(0, I).
        3. Squash: a_raw = tanh(u).
        4. Correct log-prob: log π = log N(u) − Σ log(1 − tanh²(u) + ε).
        5. Scale to [action_low, action_high].

        Parameters
        ----------
        feat:
            Latent feature tensor ``[..., feat_dim]``.

        Returns
        -------
        (action, log_prob)
            - ``action``: sampled action in [low, high], shape ``[..., action_dim]``.
            - ``log_prob``: log-probability (reparameterised), shape ``[...]``.
        """
        mean, log_std = self.forward(feat)
        std = log_std.exp()

        # Reparameterisation
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # keeps gradient

        # Squash + correct log-prob (tanh Jacobian correction)
        action_raw = torch.tanh(u)
        log_prob = (
            dist.log_prob(u) - torch.log(1.0 - action_raw.pow(2) + 1e-6)
        ).sum(dim=-1)

        action = action_raw * self.action_scale + self.action_bias
        return action, log_prob

    def deterministic_action(self, feat: torch.Tensor) -> torch.Tensor:
        """Return the mode (deterministic) action tanh(μ) rescaled."""
        mean, _ = self.forward(feat)
        return torch.tanh(mean) * self.action_scale + self.action_bias


# ──────────────────────────────────────────────────────────────────────────────
# Critic — scalar value function V(feat)
# ──────────────────────────────────────────────────────────────────────────────


class Critic(nn.Module):
    """State-value critic V(feat) for DreamerV1.

    DreamerV1 uses a *single* critic (not twin-Q); it estimates the
    expected λ-return from a given latent feature.

    Parameters
    ----------
    feat_dim:
        Dimensionality of the RSSM feature.
    hidden_dim:
        Width of the MLP hidden layers.
    """

    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Map feat[..., feat_dim] → value[..., 1]."""
        return self.net(feat)
