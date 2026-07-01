"""Analytic moment matching — the mathematical heart of PILCO.

PILCO never samples trajectories.  Instead it **propagates a Gaussian belief over
states analytically** through the learned GP dynamics.  Given an input belief

    x ~ N(mu, Sigma)

and a GP that maps inputs to outputs ``f`` (here the state delta), the true output
distribution ``p(f(x))`` is generally non-Gaussian.  PILCO approximates it by the
Gaussian with the **same first two moments** (``moment matching``) — and for the
squared-exponential kernel these moments are available in **closed form**:

* predictive mean      ``M_a   = beta_a^T q_a``
* predictive cov       ``S_ab  = beta_a^T Q^{ab} beta_b - M_a M_b (+ var term)``
* input-output cov     ``C_{:,a} = Sigma (Sigma+Lambda_a)^{-1} sum_i beta_{a,i} q_{a,i} (x_i-mu)``

The building blocks (with ``zeta_i = x_i - mu``, ``Lambda_a = diag(l_{a,d}^2)``):

    q_{a,i} = sigma_{f,a}^2 sqrt(|Lambda_a| / |Sigma + Lambda_a|)
              * exp( -1/2 zeta_i^T (Sigma + Lambda_a)^{-1} zeta_i )

    R = Sigma (Lambda_a^{-1} + Lambda_b^{-1}) + I
    Q^{ab}_{ij} = sigma_{f,a}^2 sigma_{f,b}^2 / sqrt(|R|)
                  * exp( -1/2 zeta_i^T Lambda_a^{-1} zeta_i
                         -1/2 zeta_j^T Lambda_b^{-1} zeta_j
                         +1/2 z_{ij}^T R^{-1} Sigma z_{ij} )
    with z_{ij} = Lambda_a^{-1} zeta_i + Lambda_b^{-1} zeta_j.

On the diagonal the predictive covariance also receives the GP's own (epistemic)
variance averaged over the input belief:
``S_aa += sigma_{f,a}^2 - tr((K_a + sigma_n^2 I)^{-1} Q^{aa})``.

These moments are **exact** for the SE kernel; the only approximation is replacing
the non-Gaussian output by its moment-matched Gaussian.  Crucially the whole
computation is differentiable, so gradients of a downstream cost w.r.t. the policy
flow through it by autograd (this is how PILCO optimises the policy).
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_from_scratch.pilco.gp import MultiOutputGP


def gaussian_moments(
    gp: MultiOutputGP,
    mu: Tensor,
    sigma: Tensor,
    *,
    k_invs: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Moment-match a fitted :class:`MultiOutputGP` for input ``x ~ N(mu, Sigma)``.

    Thin wrapper over :func:`moment_match` that supplies the GP's training
    inputs, per-output ``beta``, length-scales, signal variances and the cached
    ``(K + sigma_n^2 I)^{-1}`` (so the GP's own epistemic variance is included on
    the output-covariance diagonal).

    Returns ``(M[E], S[E, E], C[D, E])`` — output mean, output covariance, and
    input-output cross-covariance ``cov(x, f)``.
    """
    centers = gp.gps[0].X
    betas = [g.beta for g in gp.gps]
    lengthscales = [g.kernel.lengthscales for g in gp.gps]
    signal_var = [g.kernel.signal_variance for g in gp.gps]
    if k_invs is None:
        k_invs = [torch.cholesky_inverse(g._L) for g in gp.gps]
    return moment_match(centers, betas, lengthscales, signal_var, mu, sigma, k_invs=k_invs)


def moment_match(
    centers: Tensor,
    betas: list[Tensor],
    lengthscales: list[Tensor],
    signal_var: list[Tensor],
    mu: Tensor,
    sigma: Tensor,
    *,
    k_invs: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic moment matching for a bank of SE basis-function predictors.

    A single set of closed-form equations serves **both** the GP dynamics model
    and the RBF policy, since an RBF network is structurally a GP posterior mean.

    Parameters
    ----------
    centers:
        Shared basis input locations ``X`` (``[N, D]``) — GP training inputs or
        RBF policy centres.
    betas:
        Per-output weight vectors (each ``[N]``) — GP ``beta`` or policy weights.
    lengthscales:
        Per-output ARD length-scales (each ``[D]``).
    signal_var:
        Per-output signal variance ``sigma_f^2`` (scalar each).
    mu, sigma:
        Input belief mean ``[D]`` and covariance ``[D, D]``.
    k_invs:
        Optional per-output ``(K + sigma_n^2 I)^{-1}`` (``[N, N]``).  When given,
        the GP's epistemic predictive variance is added to the output-covariance
        diagonal.  Omit for a *deterministic* mean function (e.g. the policy).

    Returns
    -------
    ``(M[E], S[E, E], C[D, E])``.
    """
    n, d = centers.shape
    e = len(betas)
    dtype = centers.dtype
    eye = torch.eye(d, dtype=dtype)
    zeta = centers - mu                   # [N, D] centered inputs (x_i - mu)

    lambda_inv = [torch.diag(1.0 / (ls ** 2)) for ls in lengthscales]
    sf2 = signal_var

    # ------------------------------------------------------------------
    # Predictive mean M and input-output cross-covariance C
    # ------------------------------------------------------------------
    q_list: list[Tensor] = []
    mean = torch.zeros(e, dtype=dtype)
    cross = torch.zeros(d, e, dtype=dtype)
    for a in range(e):
        lam_a = torch.diag(lengthscales[a] ** 2)                  # Lambda_a
        s_plus_l = sigma + lam_a                                  # Sigma + Lambda_a
        s_plus_l_inv = torch.linalg.inv(s_plus_l)
        # c_a = sigma_f^2 * sqrt(|Lambda_a| / |Sigma + Lambda_a|)
        log_c = (
            torch.log(sf2[a])
            + 0.5 * torch.logdet(lam_a)
            - 0.5 * torch.logdet(s_plus_l)
        )
        expo = -0.5 * ((zeta @ s_plus_l_inv) * zeta).sum(dim=1)   # [N]
        q_a = torch.exp(log_c + expo)                            # [N]
        q_list.append(q_a)
        mean[a] = betas[a] @ q_a
        # C[:, a] = Sigma (Sigma+Lambda_a)^{-1} sum_i beta_i q_i (x_i - mu)
        weighted = ((betas[a] * q_a).unsqueeze(1) * zeta).sum(dim=0)   # [D]
        cross[:, a] = sigma @ s_plus_l_inv @ weighted

    # ------------------------------------------------------------------
    # Predictive covariance S
    # ------------------------------------------------------------------
    cov = torch.zeros(e, e, dtype=dtype)
    for a in range(e):
        for b in range(a, e):
            ia, ib = lambda_inv[a], lambda_inv[b]
            r = sigma @ (ia + ib) + eye                          # [D, D]
            r_inv = torch.linalg.inv(r)
            log_det_r = torch.logdet(r)

            za = zeta @ ia                                       # Lambda_a^{-1} zeta_i  [N, D]
            zb = zeta @ ib                                       # Lambda_b^{-1} zeta_j  [N, D]
            na = (za * zeta).sum(dim=1)                          # zeta_i^T ia zeta_i [N]
            nb = (zb * zeta).sum(dim=1)                          # zeta_j^T ib zeta_j [N]

            rinv_s = r_inv @ sigma                               # [D, D]
            # z_ij^T (R^{-1} Sigma) z_ij with z_ij = za_i + zb_j, expanded:
            aa = ((za @ rinv_s) * za).sum(dim=1)                 # za_i^T rinv_s za_i [N]
            cc = ((zb @ rinv_s) * zb).sum(dim=1)                 # zb_j^T rinv_s zb_j [N]
            cross_term = za @ rinv_s @ zb.t()                    # za_i^T rinv_s zb_j [N, N]
            quad = aa.unsqueeze(1) + 2.0 * cross_term + cc.unsqueeze(0)   # [N, N]

            log_q = (
                torch.log(sf2[a]) + torch.log(sf2[b]) - 0.5 * log_det_r
                - 0.5 * na.unsqueeze(1) - 0.5 * nb.unsqueeze(0) + 0.5 * quad
            )
            q_mat = torch.exp(log_q)                             # Q^{ab} [N, N]

            expected = betas[a] @ q_mat @ betas[b]               # E[f_a f_b]
            value = expected - mean[a] * mean[b]
            if a == b and k_invs is not None:
                # Add the GP's own epistemic variance, averaged over the input belief.
                value = value + sf2[a] - torch.trace(k_invs[a] @ q_mat)
            cov[a, b] = value
            cov[b, a] = value

    return mean, cov, cross
