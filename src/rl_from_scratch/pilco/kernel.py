"""Squared-Exponential (RBF) kernel with Automatic Relevance Determination.

The kernel is the heart of a Gaussian Process: it encodes the prior belief that
**nearby inputs produce nearby outputs**.  PILCO uses the squared-exponential
(a.k.a. RBF / Gaussian) kernel with **ARD** — one length-scale per input
dimension, so the model can learn that some dimensions matter more than others:

    k(x, x') = sigma_f^2 * exp( -1/2 * sum_d (x_d - x'_d)^2 / l_d^2 )

| Symbol      | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| ``l_d``     | length-scale of dimension ``d`` — how far you travel along     |
|             | ``d`` before the function value changes appreciably            |
| ``sigma_f`` | signal standard deviation — the output amplitude of the GP     |
| ``Lambda``  | ``diag(l_1^2, ..., l_D^2)`` — the ARD metric used everywhere    |
|             | in the analytic moment-matching equations                      |

Hyperparameters are stored in **log space** (``log_lengthscales``,
``log_signal_std``) so that gradient-based optimisation of the marginal
likelihood stays unconstrained while the actual values remain positive.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RBFKernel(nn.Module):
    """Squared-exponential kernel with ARD length-scales (one GP output dim).

    Parameters
    ----------
    input_dim:
        Dimension ``D`` of the kernel inputs (for PILCO: state dim + action dim).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        # Stored in log-space -> always positive after exp(), unconstrained for optim.
        self.log_lengthscales = nn.Parameter(torch.zeros(input_dim))
        self.log_signal_std = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------
    # Positive hyperparameters (exponentiate the log-space parameters)
    # ------------------------------------------------------------------
    @property
    def lengthscales(self) -> Tensor:
        """Length-scales ``l_d`` (shape ``[D]``)."""
        return torch.exp(self.log_lengthscales.clamp(-5.0, 5.0))

    @property
    def signal_variance(self) -> Tensor:
        """Signal variance ``sigma_f^2`` (scalar)."""
        return torch.exp(2.0 * self.log_signal_std.clamp(-5.0, 5.0))

    @property
    def lambda_matrix(self) -> Tensor:
        """ARD metric ``Lambda = diag(l_d^2)`` (shape ``[D, D]``)."""
        return torch.diag(self.lengthscales ** 2)

    # ------------------------------------------------------------------
    # Covariance evaluation
    # ------------------------------------------------------------------
    def forward(self, x1: Tensor, x2: Tensor | None = None) -> Tensor:
        """Covariance matrix ``K`` with entries ``k(x1_i, x2_j)``.

        Shapes: ``x1`` is ``[N, D]``, ``x2`` is ``[M, D]`` (defaults to ``x1``);
        returns ``[N, M]``.
        """
        if x2 is None:
            x2 = x1
        # Scale each dimension by 1/l_d so distances become Mahalanobis under Lambda.
        inv_l = 1.0 / self.lengthscales
        x1s = x1 * inv_l
        x2s = x2 * inv_l
        # Squared Euclidean distance in the scaled space: ||x1s||^2 + ||x2s||^2 - 2 x1s x2s^T
        sq_dist = (
            (x1s ** 2).sum(dim=1, keepdim=True)
            + (x2s ** 2).sum(dim=1).unsqueeze(0)
            - 2.0 * x1s @ x2s.t()
        )
        sq_dist = sq_dist.clamp_min(0.0)
        return self.signal_variance * torch.exp(-0.5 * sq_dist)

    def diagonal(self, x: Tensor) -> Tensor:
        """Prior variances ``k(x_i, x_i) = sigma_f^2`` (shape ``[N]``)."""
        return self.signal_variance.expand(x.shape[0])
