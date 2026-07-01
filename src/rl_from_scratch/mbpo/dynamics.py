"""Probabilistic ensemble dynamics model for MBPO.

The ensemble predicts both the delta-state and the reward jointly:

    output = [Δstate (obs_dim), reward (1)]   →  output_dim = obs_dim + 1

This allows MBPO to generate fully imagined transitions (s, a) → (s', r)
without requiring a separate hand-crafted reward function.

Key design choices (identical to PETS / Chua et al. 2018):
- SiLU activations throughout.
- Soft-bounded log-variance via learnable per-member scalar bounds.
- Bootstrap resampling per member for epistemic diversity.
- Input/output normalisation stored as registered buffers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Loss
# ======================================================================


def gaussian_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """Diagonal-Gaussian negative log-likelihood (per sample, then batch-mean).

    NLL = 0.5 * mean_batch [ sum_dim [ (target - mean)^2 * exp(-logvar) + logvar ] ]

    The constant 0.5 * log(2*pi) term is dropped (matches the PETS/MBPO formulation).

    Parameters
    ----------
    mean:
        Predicted means, shape [B, D].
    logvar:
        Predicted log-variances, shape [B, D].
    target:
        Ground-truth targets, shape [B, D].

    Returns
    -------
    Tensor
        Scalar mean NLL over the batch.
    """
    inv_var = torch.exp(-logvar)
    per_sample = 0.5 * ((target - mean) ** 2 * inv_var + logvar).sum(dim=-1)
    return per_sample.mean()


# ======================================================================
# Single probabilistic MLP
# ======================================================================


class ProbabilisticMLP(nn.Module):
    """Single probabilistic network outputting (mean, logvar) for [Δstate, reward].

    Architecture:
        Linear → SiLU → [Linear → SiLU] * (n_layers - 1) → Linear(2 * output_dim)

    The output is split into ``mean`` and ``logvar``.
    Log-variance is soft-bounded by learnable scalar parameters ``max_logvar``
    and ``min_logvar``:

        logvar = max_logvar - softplus(max_logvar - raw_logvar)
        logvar = min_logvar + softplus(logvar - min_logvar)

    Parameters
    ----------
    input_dim:
        Input dimensionality (obs_dim + act_dim).
    output_dim:
        Output dimensionality (obs_dim + 1 — predicts delta state AND reward).
    hidden_dim:
        Width of each hidden layer.
    n_layers:
        Number of hidden layers (>= 1).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 200,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 2 * output_dim))

        self.net = nn.Sequential(*layers)

        # Learnable soft logvar bounds (per output dim)
        self.max_logvar = nn.Parameter(
            torch.full((output_dim,), 0.5, dtype=torch.float32)
        )
        self.min_logvar = nn.Parameter(
            torch.full((output_dim,), -10.0, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning (mean, logvar), each shape [B, output_dim].

        Parameters
        ----------
        x:
            Input tensor, shape [B, input_dim].

        Returns
        -------
        mean:
            Predicted [Δstate, reward] mean, shape [B, output_dim].
        logvar:
            Predicted log-variance (soft-bounded), shape [B, output_dim].
        """
        out = self.net(x)
        raw_mean, raw_logvar = out.chunk(2, dim=-1)

        # Soft bound: max_logvar - softplus(max_logvar - raw_logvar)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        # Soft bound: min_logvar + softplus(logvar - min_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return raw_mean, logvar


# ======================================================================
# Probabilistic ensemble
# ======================================================================


class ProbabilisticEnsemble(nn.Module):
    """Ensemble of ``ensemble_size`` independent probabilistic MLPs.

    Each member jointly models:
        [Δstate (obs_dim), reward (1)]  →  output_dim = obs_dim + 1

    Input and output normalisation statistics are computed once and stored
    as registered buffers for save/load compatibility.

    Parameters
    ----------
    input_dim:
        obs_dim + act_dim.
    output_dim:
        obs_dim + 1 (predicts delta state AND reward).
    ensemble_size:
        Number of independent ensemble members.
    hidden_dim:
        Hidden layer width for each member.
    n_layers:
        Number of hidden layers for each member.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 7,
        hidden_dim: int = 200,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.members = nn.ModuleList([
            ProbabilisticMLP(input_dim, output_dim, hidden_dim=hidden_dim, n_layers=n_layers)
            for _ in range(ensemble_size)
        ])

        # Normalisation statistics — x is [obs, act], y is [Δstate, reward]
        self.register_buffer("x_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(input_dim, dtype=torch.float32))
        self.register_buffer("y_mean", torch.zeros(output_dim, dtype=torch.float32))
        self.register_buffer("y_std", torch.ones(output_dim, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def fit_normalizer(self, X: Tensor, Y: Tensor) -> None:
        """Compute and store input/output normalisation statistics.

        Parameters
        ----------
        X:
            Training inputs, shape [N, input_dim].  Columns: [obs, act].
        Y:
            Training targets, shape [N, output_dim].  Columns: [Δstate, reward].
        """
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp_min(1e-6))
        self.y_mean.copy_(Y.mean(0))
        self.y_std.copy_(Y.std(0).clamp_min(1e-6))

    def _normalize_x(self, X: Tensor) -> Tensor:
        return (X - self.x_mean) / self.x_std

    def _normalize_y(self, Y: Tensor) -> Tensor:
        return (Y - self.y_mean) / self.y_std

    def _denormalize_y(self, Y_norm: Tensor) -> Tensor:
        return Y_norm * self.y_std + self.y_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Tensor,
        Y: Tensor,
        *,
        steps: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
    ) -> float:
        """Train all ensemble members with bootstrap resampling.

        Each member trains on an independent bootstrap resample of the full
        dataset for ``steps`` Adam steps.  Logvar regularisation is added:

            reg = 0.01 * (max_logvar.sum() - min_logvar.sum())

        Parameters
        ----------
        X:
            Training inputs [s, a], shape [N, input_dim].
        Y:
            Training targets [Δs, r], shape [N, output_dim].
        steps:
            Number of Adam gradient steps per ensemble member.
        batch_size:
            Mini-batch size per step.
        lr:
            Adam learning rate.
        weight_decay:
            L2 regularisation coefficient.

        Returns
        -------
        float
            Mean final NLL over all members (normalised space).
        """
        self.fit_normalizer(X, Y)

        X_norm = self._normalize_x(X)
        Y_norm = self._normalize_y(Y)

        N = X_norm.shape[0]
        final_nlls: list[float] = []

        for member in self.members:
            member.train()
            optimizer = torch.optim.Adam(
                member.parameters(), lr=lr, weight_decay=weight_decay
            )

            # Bootstrap resample: sample N indices with replacement
            idx = torch.randint(0, N, (N,))
            X_boot = X_norm[idx]
            Y_boot = Y_norm[idx]

            last_nll = 0.0
            for _ in range(steps):
                batch_idx = torch.randint(0, N, (min(batch_size, N),))
                x_batch = X_boot[batch_idx]
                y_batch = Y_boot[batch_idx]

                optimizer.zero_grad()
                mean, logvar = member(x_batch)
                nll = gaussian_nll(mean, logvar, y_batch)
                reg = 0.01 * (member.max_logvar.sum() - member.min_logvar.sum())
                loss = nll + reg
                loss.backward()
                optimizer.step()
                last_nll = float(nll.item())

            member.eval()
            final_nlls.append(last_nll)

        mean_nll = float(np.mean(final_nlls))
        return mean_nll if math.isfinite(mean_nll) else 0.0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict [Δstate, reward] (mean, variance) from all members.

        Parameters
        ----------
        x:
            Raw (un-normalised) inputs, shape [B, input_dim].

        Returns
        -------
        means:
            De-normalised means from all members, shape [B, ensemble_size, output_dim].
        vars_:
            De-normalised variances from all members, shape [B, ensemble_size, output_dim].
        """
        x_norm = self._normalize_x(x)
        all_means: list[Tensor] = []
        all_vars: list[Tensor] = []

        for member in self.members:
            member.eval()
            mean_norm, logvar = member(x_norm)
            mean_denorm = mean_norm * self.y_std + self.y_mean
            var_denorm = torch.exp(logvar) * (self.y_std ** 2)
            all_means.append(mean_denorm)
            all_vars.append(var_denorm)

        means = torch.stack(all_means, dim=1)   # [B, E, D]
        vars_ = torch.stack(all_vars, dim=1)    # [B, E, D]
        return means, vars_

    def propagate(
        self,
        states: Tensor,
        actions: Tensor,
        model_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One-step dynamics + reward propagation with stochastic sampling.

        For each row i, selects member ``model_idx[i]`` and samples
        [Δstate, reward] from the diagonal Gaussian in normalised space,
        then de-normalises.

        Parameters
        ----------
        states:
            Current states, shape [B, obs_dim].
        actions:
            Actions, shape [B, act_dim].
        model_idx:
            Integer member indices per row, shape [B].

        Returns
        -------
        next_states:
            Predicted next states, shape [B, obs_dim].
        rewards:
            Predicted scalar rewards, shape [B].
        """
        B = states.shape[0]
        obs_dim = states.shape[1]
        x = torch.cat([states, actions], dim=-1)   # [B, input_dim]
        x_norm = self._normalize_x(x)              # [B, input_dim]

        means_norm = torch.zeros(B, self.output_dim, dtype=x.dtype, device=x.device)
        logvars_norm = torch.zeros_like(means_norm)

        # Dispatch each row to its assigned member
        for m_idx, member in enumerate(self.members):
            mask = (model_idx == m_idx)
            if not mask.any():
                continue
            member.eval()
            m_n, lv = member(x_norm[mask])
            means_norm[mask] = m_n
            logvars_norm[mask] = lv

        # Sample [Δstate, reward] in normalised space, then de-normalise
        eps = torch.randn_like(means_norm)
        std_norm = torch.exp(0.5 * logvars_norm)
        sample_norm = means_norm + std_norm * eps
        # De-normalise: multiply by y_std and add y_mean
        sample = sample_norm * self.y_std + self.y_mean

        # Split: first obs_dim columns are Δstate, last column is reward
        delta = sample[:, :obs_dim]
        reward = sample[:, obs_dim]

        next_states = states + delta
        return next_states, reward

    @torch.no_grad()
    def disagreement(self, X: Tensor) -> float:
        """Epistemic uncertainty: mean inter-member std of the means.

        Parameters
        ----------
        X:
            Raw inputs, shape [B, input_dim].

        Returns
        -------
        float
            Mean (over batch and dimensions) of the inter-member std.
        """
        means, _ = self.predict(X)       # [B, E, D]
        epistemic_std = means.std(dim=1) # [B, D]
        return float(epistemic_std.mean().item())
