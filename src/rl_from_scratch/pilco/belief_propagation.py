"""Analytic belief propagation helpers for PILCO.

This module contains the Gaussian moment-matching step used by classic PILCO:
propagate a Gaussian belief through the GP dynamics under a policy, then
accumulate the expected trajectory cost.  Deep PILCO's particle propagation
lives in ``bnn.py`` because it is part of the BNN/dropout model variant.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from rl_from_scratch.pilco.cost import expected_cost
from rl_from_scratch.pilco.gp import MultiOutputGP
from rl_from_scratch.pilco.moment_matching import gaussian_moments


def _psd_project(sigma: Tensor, floor: float = 1e-6) -> Tensor:
    sigma = 0.5 * (sigma + sigma.t())
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    eye = torch.eye(sigma.shape[-1], dtype=sigma.dtype, device=sigma.device)
    with torch.no_grad():
        try:
            lam_min = torch.linalg.eigvalsh(sigma).min()
        except torch.linalg.LinAlgError:
            lam_min = torch.tensor(-1.0, dtype=sigma.dtype, device=sigma.device)
    return sigma + torch.clamp(torch.tensor(floor, dtype=sigma.dtype, device=sigma.device) - lam_min, min=0.0) * eye


def project_encoded_angle_torch(x: Tensor) -> Tensor:
    """Project ``[..., 5]`` encoded InvertedPendulum states to sin²+cos²=1."""
    sin_theta = x[..., 1]
    cos_theta = x[..., 2]
    norm = torch.sqrt((sin_theta * sin_theta + cos_theta * cos_theta).clamp_min(1e-8))
    parts = [
        x[..., :1],
        (sin_theta / norm).unsqueeze(-1),
        (cos_theta / norm).unsqueeze(-1),
    ]
    if x.shape[-1] > 3:
        parts.append(x[..., 3:])
    return torch.cat(parts, dim=-1)


def project_angle_belief_torch(mu: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Project encoded-state mean and covariance via a Jacobian linearisation."""
    mu_proj = project_encoded_angle_torch(mu)
    sin_theta = mu[1]
    cos_theta = mu[2]
    radius_sq = (sin_theta * sin_theta + cos_theta * cos_theta).clamp_min(1e-8)
    radius = torch.sqrt(radius_sq)
    radius_cubed = radius_sq * radius

    jac = torch.eye(mu.shape[0], dtype=mu.dtype, device=mu.device)
    jac[1, 1] = (cos_theta * cos_theta) / radius_cubed
    jac[1, 2] = -(sin_theta * cos_theta) / radius_cubed
    jac[2, 1] = jac[1, 2]
    jac[2, 2] = (sin_theta * sin_theta) / radius_cubed

    sigma_proj = _psd_project(jac @ sigma @ jac.t())
    return mu_proj, sigma_proj


def propagate(
    gp: MultiOutputGP,
    policy: nn.Module,
    mu_x: Tensor,
    sigma_x: Tensor,
    *,
    project_encoded_angle: bool = False,
    k_invs: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Propagate one Gaussian belief step under the GP dynamics."""
    state_dim = mu_x.shape[0]
    mu_u, sigma_u, c_xu = policy.propagate(mu_x, sigma_x)
    mu_joint = torch.cat([mu_x, mu_u])
    top = torch.cat([sigma_x, c_xu], dim=1)
    bottom = torch.cat([c_xu.t(), sigma_u], dim=1)
    sigma_joint = _psd_project(torch.cat([top, bottom], dim=0))

    delta_mean, delta_cov, c_joint_delta = gaussian_moments(
        gp,
        mu_joint,
        sigma_joint,
        k_invs=k_invs,
    )
    c_x_delta = c_joint_delta[:state_dim, :]

    mu_next = mu_x + delta_mean
    sigma_next = _psd_project(sigma_x + delta_cov + c_x_delta + c_x_delta.t())

    if project_encoded_angle:
        mu_next, sigma_next = project_angle_belief_torch(mu_next, sigma_next)

    return mu_next, sigma_next


def predict_trajectory(
    gp: MultiOutputGP,
    policy: nn.Module,
    mu0: Tensor,
    sigma0: Tensor,
    *,
    horizon: int,
    target: Tensor,
    weight: Tensor,
    project_encoded_angle: bool = False,
    step_cost_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    k_invs: list[Tensor] | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """Roll the belief forward and accumulate either generic or custom cost."""
    mu, sigma = mu0, sigma0
    total = torch.zeros((), dtype=mu0.dtype, device=mu0.device)
    means = [mu]
    for _ in range(horizon):
        mu_u, sigma_u, _ = policy.propagate(mu, sigma)
        mu, sigma = propagate(
            gp,
            policy,
            mu,
            sigma,
            project_encoded_angle=project_encoded_angle,
            k_invs=k_invs,
        )
        if step_cost_fn is None:
            total = total + expected_cost(mu, sigma, target, weight)
        else:
            total = total + step_cost_fn(mu, sigma, mu_u, sigma_u)
        means.append(mu)
    return total, means
