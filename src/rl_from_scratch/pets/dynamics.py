"""Probabilistic ensemble dynamics model for PETS.

The ensemble consists of ``ensemble_size`` independent probabilistic MLPs,
each modelling the delta-dynamics distribution:

    p(s_{t+1} - s_t | s_t, a_t) = N(mu(s_t, a_t), diag(sigma^2(s_t, a_t)))

Key design choices (from Chua et al. 2018):
- SiLU activations throughout (smooth, bounded derivative).
- Soft-bounded log-variance via learnable per-member scalar bounds
  (max_logvar, min_logvar) following the PETS paper formulation.
- Bootstrap resampling: each member trains on its own resample of the data,
  introducing diversity that gives a meaningful epistemic uncertainty signal.
- Input/output normalisation: computed once from the full dataset and stored
  as registered buffers.  Members predict in normalised space; predictions are
  de-normalised back to the original space before returning.
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

    The constant 0.5 * log(2*pi) term is dropped (matches the PETS formulation).

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
    """Single probabilistic dynamics network outputting (mean, logvar).

    Architecture:
        Linear → SiLU → [Linear → SiLU] * (n_layers - 1) → Linear(2 * output_dim)

    The output is split into ``mean`` (first half) and ``logvar`` (second half).
    Log-variance is soft-bounded by learnable scalar parameters ``max_logvar``
    and ``min_logvar``:

        logvar = max_logvar - softplus(max_logvar - raw_logvar)
        logvar = min_logvar + softplus(logvar - min_logvar)

    Parameters
    ----------
    input_dim:
        Input dimensionality (obs_dim + act_dim).
    output_dim:
        Output dimensionality (obs_dim — predicts delta state).
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
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Build the hidden layers
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.SiLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 2 * output_dim))

        self.net = nn.Sequential(*layers)

        # Learnable soft logvar bounds (per output dim, initialised as scalars
        # broadcast over all output dimensions)
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
            Predicted delta-state mean, shape [B, output_dim].
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

    Each member has its own random initialisation and trains on an independent
    bootstrap resample of the data.  This diversity is the source of epistemic
    uncertainty.

    Input and output normalisation statistics (mean/std) are computed once
    from the full training set and stored as registered buffers so they are
    saved/loaded with the model state.

    Parameters
    ----------
    input_dim:
        obs_dim + act_dim.
    output_dim:
        obs_dim (predicts delta state).
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
        ensemble_size: int = 5,
        hidden_dim: int = 200,
        n_layers: int = 3,
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

        # Normalisation statistics (float32 buffers)
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
            Training inputs, shape [N, input_dim].
        Y:
            Training targets (delta states), shape [N, output_dim].
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

        Each member trains on an independent bootstrap resample (sample with
        replacement) of the full dataset for ``steps`` Adam steps.  The
        logvar regularisation term from the PETS paper is added to each
        member's loss:

            reg = 0.01 * (max_logvar.sum() - min_logvar.sum())

        Parameters
        ----------
        X:
            Training inputs, shape [N, input_dim].
        Y:
            Training targets, shape [N, output_dim].
        steps:
            Number of Adam gradient steps per ensemble member.
        batch_size:
            Mini-batch size drawn from each member's bootstrap resample.
        lr:
            Adam learning rate.
        weight_decay:
            L2 regularisation coefficient for Adam.

        Returns
        -------
        float
            Mean final NLL over all members (in the normalised space).
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
                # Sample a mini-batch from the bootstrap resample
                batch_idx = torch.randint(0, N, (min(batch_size, N),))
                x_batch = X_boot[batch_idx]
                y_batch = Y_boot[batch_idx]

                optimizer.zero_grad()
                mean, logvar = member(x_batch)
                nll = gaussian_nll(mean, logvar, y_batch)
                # Logvar regularisation (keeps bounds from collapsing to ±inf)
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
        """Predict delta-state (mean, variance) from all members.

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
            # De-normalise: mean and std scale independently
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
    ) -> Tensor:
        """One-step dynamics propagation with stochastic sampling.

        For each row i, selects member ``model_idx[i]`` and samples the
        next state from the diagonal Gaussian:

            next_state[i] = state[i] + delta[i]
            delta[i] ~ N(mean_m(state[i], action[i]), diag(var_m(state[i], action[i])))

        Parameters
        ----------
        states:
            Current states, shape [B, obs_dim].
        actions:
            Actions, shape [B, act_dim].
        model_idx:
            Integer member indices to use per row, shape [B].

        Returns
        -------
        Tensor
            Next states, shape [B, obs_dim].
        """
        B = states.shape[0]
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

        # Sample delta in normalised space, then de-normalise
        eps = torch.randn_like(means_norm)
        std_norm = torch.exp(0.5 * logvars_norm)
        delta_norm = means_norm + std_norm * eps
        delta = delta_norm * self.y_std + self.y_mean

        return states + delta

    @torch.no_grad()
    def disagreement(self, X: Tensor) -> float:
        """Epistemic uncertainty: mean per-dim std of member means over a batch.

        Parameters
        ----------
        X:
            Raw inputs, shape [B, input_dim].

        Returns
        -------
        float
            Mean (over batch and dimensions) of the inter-member std.
        """
        means, _ = self.predict(X)           # [B, E, D]
        # Std over ensemble members for each (batch, dim) pair
        epistemic_std = means.std(dim=1)     # [B, D]
        return float(epistemic_std.mean().item())
