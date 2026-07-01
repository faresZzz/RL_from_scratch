"""Cost functions for PILCO and Deep PILCO.

The generic PILCO objective uses a bounded saturating state cost.  For the Gym
``InvertedPendulum-v5`` stabilisation task we additionally expose a denser,
task-aware objective that combines:

1. state deviation from the upright encoded target ``[0, 0, 1, 0, 0]``;
2. action effort;
3. a smooth approximation of termination risk near the cart/angle limits.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def saturating_cost(x: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Pointwise saturating cost ``c(x) in [0, 1]`` for deterministic states."""
    delta = x - target
    quad = (delta @ weight * delta).sum(dim=-1)
    return 1.0 - torch.exp(-0.5 * quad)


def expected_cost(mu: Tensor, sigma: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Closed-form expected saturating cost ``E[c(x)]`` for ``x ~ N(mu, Sigma)``."""
    d = mu.shape[0]
    eye = torch.eye(d, dtype=mu.dtype, device=mu.device)
    sigma = 0.5 * (sigma + sigma.t())
    delta = mu - target
    # ponytail: assumes a diagonal cost weight (only the diagonal of `weight` is
    # used). Every caller passes diag(W); add a full matrix sqrt if an
    # off-diagonal cost is ever needed.
    sqrt_w = torch.diag(torch.sqrt(torch.clamp(torch.diagonal(weight), min=0.0)))
    a = eye + sqrt_w @ sigma @ sqrt_w
    a = 0.5 * (a + a.t()) + 1e-8 * eye
    inner = sqrt_w @ torch.linalg.solve(a, sqrt_w)
    quad = delta @ inner @ delta
    log_det = torch.linalg.slogdet(a).logabsdet
    expectation = torch.exp(-0.5 * log_det - 0.5 * quad).clamp(0.0, 1.0)
    return (1.0 - expectation).clamp(0.0, 1.0)


def _default_ip_target(state_dim: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if state_dim == 5:
        return torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=dtype, device=device)
    if state_dim == 4:
        return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    return torch.zeros(state_dim, dtype=dtype, device=device)

def expected_inverted_pendulum_cost(
    mu_x: Tensor,
    sigma_x: Tensor,
    mu_u: Tensor,
    sigma_u: Tensor,
    *,
    state_weight: Tensor | None = None,
    action_weight: Tensor | None = None,
    target: Tensor | None = None,
    terminal_penalty: float = 10.0,
    action_cost_weight: float = 1e-4,
) -> Tensor:
    """PILCO cost: bounded task loss plus analytic Gym failure probability."""
    if target is None:
        target = _default_ip_target(mu_x.shape[0], dtype=mu_x.dtype, device=mu_x.device)
    if state_weight is None:
        state_weight = torch.diag(
            torch.tensor([0.01, 100.0, 1.0, 0.01, 2.0], dtype=mu_x.dtype, device=mu_x.device)[: mu_x.shape[0]]
        )
    del action_weight

    # Original PILCO keeps the state objective bounded; uncertainty enters
    # analytically through E[c(x)] rather than through a point estimate.
    state_term = expected_cost(mu_x, sigma_x, target, state_weight)
    action_term = action_cost_weight * (
        mu_u.square().sum() + torch.diagonal(sigma_u).sum().clamp_min(0.0)
    )

    # Around upright, sin(theta) is monotonic. Its Gaussian marginal gives a
    # closed-form approximation of P(|theta| > 0.2), Gym's actual failure rule.
    threshold = math.sin(0.2)
    std_sin = torch.sqrt(sigma_x[1, 1].clamp_min(1e-10))
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=mu_x.dtype, device=mu_x.device),
        torch.ones((), dtype=mu_x.dtype, device=mu_x.device),
    )
    safe_probability = normal.cdf((threshold - mu_x[1]) / std_sin) - normal.cdf(
        (-threshold - mu_x[1]) / std_sin
    )
    risk = (1.0 - safe_probability).clamp(0.0, 1.0)
    return state_term + action_term + terminal_penalty * risk


def inverted_pendulum_particle_cost(
    states: Tensor,
    actions: Tensor,
    *,
    state_weight: Tensor | None = None,
    action_weight: Tensor | None = None,
    target: Tensor | None = None,
    terminal_penalty: float = 150.0,
    risk_sharpness: float = 10.0,
) -> Tensor:
    """Dense differentiable particle cost used by the successful IP demo."""
    del state_weight, action_weight, target, risk_sharpness
    theta = torch.atan2(states[..., 1], states[..., 2])
    state_term = (
        0.5 * states[..., 0].square()
        + 35.0 * theta.square()
        + 0.05 * states[..., 3].square()
        + 0.7 * states[..., 4].square()
    )
    action_term = 0.01 * actions.square().sum(dim=-1)
    boundary = terminal_penalty * torch.relu(theta.abs() - 0.14).square()
    return state_term + action_term + boundary
