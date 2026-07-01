"""MLP building blocks for DreamerV1.

All networks use SiLU activations (smoother than ReLU, works well in
latent-space models).  ``feat_dim = deter_dim + stoch_dim``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Observation encoder / decoder
# ──────────────────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    """Encode a flat observation into a compact embedding.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the raw observation.
    hidden_dim:
        Width of the single hidden layer.
    embed_dim:
        Output embedding dimensionality.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map obs[..., obs_dim] → embed[..., embed_dim]."""
        return self.net(obs)


class Decoder(nn.Module):
    """Decode a latent feature vector back to an observation mean.

    The observation reconstruction loss is MSE (Gaussian with unit variance).

    Parameters
    ----------
    feat_dim:
        Dimensionality of the feature (deter_dim + stoch_dim).
    hidden_dim:
        Width of the single hidden layer.
    obs_dim:
        Dimensionality of the reconstructed observation.
    """

    def __init__(self, feat_dim: int, hidden_dim: int, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Map feat[..., feat_dim] → obs_mean[..., obs_dim]."""
        return self.net(feat)


# ──────────────────────────────────────────────────────────────────────────────
# Reward model
# ──────────────────────────────────────────────────────────────────────────────


class RewardModel(nn.Module):
    """Predict a scalar reward from the latent feature vector.

    Parameters
    ----------
    feat_dim:
        Dimensionality of the feature (deter_dim + stoch_dim).
    hidden_dim:
        Width of the single hidden layer.
    """

    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Map feat[..., feat_dim] → reward[..., 1]."""
        return self.net(feat)
