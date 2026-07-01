"""Gaussian Process regression from scratch (the PILCO dynamics model).

A Gaussian Process places a distribution over functions.  After conditioning on
observed transitions, it returns for any query input **both** a predictive mean
**and** a predictive variance — the calibrated uncertainty that makes PILCO so
sample-efficient.

For a single output, with inputs ``X`` (``N x D``), targets ``y`` (``N``),
kernel matrix ``K``, and observation noise ``sigma_n^2``:

* **Training** maximises the log marginal likelihood

      log p(y | X) = -1/2 y^T (K + sigma_n^2 I)^{-1} y
                     - 1/2 log|K + sigma_n^2 I| - N/2 log(2 pi)

  This automatically balances data-fit against model complexity (Occam's razor)
  and learns the kernel length-scales, the signal variance and the noise.

* **Prediction** at a test point ``x*`` gives

      mean(x*) = k_*^T (K + sigma_n^2 I)^{-1} y = k_*^T beta
      var(x*)  = k(x*, x*) - k_*^T (K + sigma_n^2 I)^{-1} k_*

  where ``beta = (K + sigma_n^2 I)^{-1} y`` is cached after training and reused
  by the analytic moment-matching equations.

PILCO models the dynamics with **one independent GP per output dimension**
(predicting the state *delta* ``x_{t+1} - x_t``); ``MultiOutputGP`` bundles them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from rl_from_scratch.pilco.kernel import RBFKernel

# Small diagonal jitter added before Cholesky for numerical stability.
_JITTER = 1e-6


def _robust_cholesky(matrix: Tensor) -> Tensor:
    """Cholesky factor with adaptive jitter.

    During hyperparameter optimisation the kernel matrix can become numerically
    non-positive-definite (near-duplicate inputs, extreme length-scales).  We
    retry with geometrically increasing diagonal jitter — the standard GP trick —
    instead of crashing.
    """
    n = matrix.shape[-1]
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    jitter = _JITTER
    for _ in range(8):
        try:
            return torch.linalg.cholesky(matrix + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10.0
    # Last resort: factor the strongly-regularised matrix.
    return torch.linalg.cholesky(matrix + jitter * eye)


class GaussianProcess(nn.Module):
    """Single-output GP regressor with an SE-ARD kernel and bounded noise.

    The observation noise is learned by marginal likelihood, as in PILCO. Its
    log-standard-deviation is clamped during use so deterministic simulator
    deltas can select tiny noise without collapsing Cholesky numerics.

    Kernel and noise parameters are kept in a numerically safe log range. This
    retains the marginal-likelihood objective used by PILCO without a prior that
    would pull small, data-scaled hyperparameters back toward one.
    """

    def __init__(self, input_dim: int, fixed_noise_var: float | None = None) -> None:
        super().__init__()
        self.kernel = RBFKernel(input_dim)
        initial_noise_var = 1e-4 if fixed_noise_var is None else float(fixed_noise_var)
        self.log_noise_std = nn.Parameter(
            torch.tensor(0.5 * math.log(initial_noise_var), dtype=torch.float64),
            requires_grad=fixed_noise_var is None,
        )
        # Data + cached training-time quantities (filled by set_data / _refresh_cache).
        self.register_buffer("X", torch.zeros(0, input_dim))
        self.register_buffer("y", torch.zeros(0))
        self.register_buffer("_beta", torch.zeros(0))
        self.register_buffer("_L", torch.zeros(0, 0))

    @property
    def noise_variance(self) -> Tensor:
        """Bounded learned observation-noise variance ``sigma_n^2``."""
        log_std = self.log_noise_std.clamp(-9.0, 1.0)
        return torch.exp(2.0 * log_std) + 1e-8

    @property
    def beta(self) -> Tensor:
        """Cached ``beta = (K + sigma_n^2 I)^{-1} y`` (shape ``[N]``)."""
        return self._beta

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def set_data(self, X: Tensor, y: Tensor) -> None:
        """Attach training inputs ``X`` (``[N, D]``) and targets ``y`` (``[N]``)."""
        self.X = X.detach().clone()
        self.y = y.detach().clone()

    def initialize_from_data(self, X: Tensor, y: Tensor) -> None:
        """Set a data-scale-aware starting point before the first L-BFGS fit."""
        with torch.no_grad():
            x_scale = torch.nan_to_num(X.std(dim=0, unbiased=False), nan=1.0).clamp_min(1e-2)
            y_scale = torch.nan_to_num(y.std(unbiased=False), nan=torch.tensor(1.0, dtype=y.dtype, device=y.device)).clamp_min(1e-3)
            self.kernel.log_lengthscales.copy_(torch.log(x_scale))
            self.kernel.log_signal_std.copy_(torch.log(y_scale))
            if self.log_noise_std.requires_grad:
                self.log_noise_std.copy_(torch.log((0.05 * y_scale).clamp_min(1e-4)))

    def _noisy_gram(self) -> Tensor:
        """``K(X, X) + (sigma_n^2 + jitter) I`` — the matrix to invert."""
        n = self.X.shape[0]
        gram = self.kernel(self.X)
        eye = torch.eye(n, dtype=gram.dtype, device=gram.device)
        return gram + (self.noise_variance + _JITTER) * eye

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------
    def negative_log_marginal_likelihood(self) -> Tensor:
        """Negative log marginal likelihood."""
        n = self.X.shape[0]
        k_noisy = self._noisy_gram()
        chol = _robust_cholesky(k_noisy)
        alpha = torch.cholesky_solve(self.y.unsqueeze(1), chol).squeeze(1)
        # log|K| = 2 sum(log(diag(L)));  data term = 1/2 y^T alpha.
        data_fit = 0.5 * (self.y * alpha).sum()
        log_det = torch.log(torch.diagonal(chol)).sum()
        const = 0.5 * n * math.log(2.0 * math.pi)
        nlml = data_fit + log_det + const
        return nlml

    def fit(
        self,
        X: Tensor | None = None,
        y: Tensor | None = None,
        *,
        n_steps: int = 50,
        lr: float = 1.0,
    ) -> float:
        """Optimise the hyperparameters by minimising the NLML (+ prior) with L-BFGS.

        If ``X`` and ``y`` are given they are attached first (otherwise the data
        previously passed to :meth:`set_data` is used).  Returns the final NLML
        value; the predictive cache (``beta`` and the Cholesky factor) is then
        refreshed.
        """
        if X is not None and y is not None:
            self.set_data(X, y)
        optimizer = torch.optim.LBFGS(
            self.parameters(), lr=lr, max_iter=n_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = self.negative_log_marginal_likelihood()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            final = self.negative_log_marginal_likelihood()
        self._refresh_cache()
        return float(final)

    def _refresh_cache(self) -> None:
        """Recompute and cache the Cholesky factor and ``beta`` after training."""
        with torch.no_grad():
            k_noisy = self._noisy_gram()
            chol = _robust_cholesky(k_noisy)
            self._L = chol
            self._beta = torch.cholesky_solve(self.y.unsqueeze(1), chol).squeeze(1)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, x_star: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and variance at query points ``x_star`` (``[M, D]``).

        Returns ``(mean[M], var[M])``.
        """
        k_star = self.kernel(self.X, x_star)            # [N, M]
        mean = k_star.t() @ self._beta                  # [M]
        # var = k** - k_*^T (K+sigma^2 I)^{-1} k_*  via the cached Cholesky factor.
        v = torch.cholesky_solve(k_star, self._L)       # [N, M]
        reduction = (k_star * v).sum(dim=0)             # [M]
        prior_var = self.kernel.diagonal(x_star)        # [M]
        var = (prior_var - reduction).clamp_min(0.0)
        return mean, var


class MultiOutputGP(nn.Module):
    """A bank of independent single-output GPs sharing the same input space.

    PILCO predicts the state delta ``Delta x = x_{t+1} - x_t``; each output
    dimension gets its own GP (its own length-scales / noise), as in the
    original method.
    """

    def __init__(self, input_dim: int, output_dim: int, fixed_noise_var: float | None = None) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.gps = nn.ModuleList(
            GaussianProcess(input_dim, fixed_noise_var=fixed_noise_var)
            for _ in range(output_dim)
        )

    def set_data(self, X: Tensor, Y: Tensor) -> None:
        """Attach shared inputs ``X`` (``[N, D]``) and targets ``Y`` (``[N, E]``)."""
        for e, gp in enumerate(self.gps):
            gp.set_data(X, Y[:, e])

    def initialize_from_data(self, X: Tensor, Y: Tensor) -> None:
        for e, gp in enumerate(self.gps):
            gp.initialize_from_data(X, Y[:, e])

    def fit(
        self,
        X: Tensor | None = None,
        Y: Tensor | None = None,
        *,
        n_steps: int = 50,
        lr: float = 1.0,
    ) -> list[float]:
        """Fit every output GP; returns the per-output final NLML.

        If ``X`` and ``Y`` are given they are attached first (``Y`` is ``[N, E]``).
        """
        if X is not None and Y is not None:
            self.set_data(X, Y)
        return [gp.fit(n_steps=n_steps, lr=lr) for gp in self.gps]

    def predict(self, x_star: Tensor) -> tuple[Tensor, Tensor]:
        """Stacked predictive means/vars: ``(mean[M, E], var[M, E])``."""
        means, varis = [], []
        for gp in self.gps:
            m, v = gp.predict(x_star)
            means.append(m)
            varis.append(v)
        return torch.stack(means, dim=1), torch.stack(varis, dim=1)
