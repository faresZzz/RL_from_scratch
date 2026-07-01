"""Policies for PILCO and Deep PILCO."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rl_from_scratch.pilco.moment_matching import moment_match


def _sine_squash_moments(
    m_raw: Tensor,
    s_raw: Tensor,
    c_raw: Tensor,
    action_high: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Push Gaussian raw-control moments through ``u_max * sin(.)``."""
    diag_v = torch.diagonal(s_raw)
    exp_half = torch.exp(-0.5 * diag_v)
    mu_u = action_high * exp_half * torch.sin(m_raw)
    var_u = action_high ** 2 * (
        0.5 * (1.0 - torch.exp(-2.0 * diag_v) * torch.cos(2.0 * m_raw))
        - torch.exp(-diag_v) * torch.sin(m_raw) ** 2
    )
    gain = action_high * exp_half * torch.cos(m_raw)
    sigma_u = torch.outer(gain, gain) * s_raw
    sigma_u = sigma_u - torch.diag(torch.diagonal(sigma_u)) + torch.diag(var_u)
    c_xu = c_raw * gain.unsqueeze(0)
    return mu_u, sigma_u, c_xu


class RBFPolicy(nn.Module):
    """Deterministic RBF policy with sine action-saturation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        n_basis: int = 50,
        action_high: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_basis = int(n_basis)
        self.centers = nn.Parameter(torch.randn(n_basis, state_dim))
        self.weights = nn.Parameter(0.1 * torch.randn(n_basis, action_dim))
        self.log_lengthscales = nn.Parameter(torch.zeros(state_dim))
        if action_high is None:
            action_high = torch.ones(action_dim)
        self.register_buffer("action_high", torch.as_tensor(action_high).flatten())

    @property
    def lengthscales(self) -> Tensor:
        return torch.exp(self.log_lengthscales)

    def raw_control(self, x: Tensor) -> Tensor:
        single = x.dim() == 1
        xb = x.unsqueeze(0) if single else x
        inv_l = 1.0 / self.lengthscales
        diff = (xb.unsqueeze(1) - self.centers.unsqueeze(0)) * inv_l
        phi = torch.exp(-0.5 * (diff ** 2).sum(dim=2))
        out = phi @ self.weights
        return out.squeeze(0) if single else out

    def forward(self, x: Tensor) -> Tensor:
        return self.action_high * torch.sin(self.raw_control(x))

    def propagate(self, mu_x: Tensor, sigma_x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        du = self.action_dim
        betas = [self.weights[:, k] for k in range(du)]
        lengthscales = [self.lengthscales for _ in range(du)]
        signal_var = [torch.ones((), dtype=mu_x.dtype, device=mu_x.device) for _ in range(du)]
        m_raw, s_raw, c_raw = moment_match(
            self.centers,
            betas,
            lengthscales,
            signal_var,
            mu_x,
            sigma_x,
            k_invs=None,
        )
        return _sine_squash_moments(m_raw, s_raw, c_raw, self.action_high)


class LinearSinePolicy(nn.Module):
    """Linear pre-activation policy with the same sine saturation as PILCO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        action_high: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        # Start from the zero-action baseline; PILCO then learns the feedback
        # gain from model gradients instead of inheriting a random controller.
        self.weight = nn.Parameter(torch.zeros(state_dim, action_dim))
        self.bias = nn.Parameter(torch.zeros(action_dim))
        if action_high is None:
            action_high = torch.ones(action_dim)
        self.register_buffer("action_high", torch.as_tensor(action_high).flatten())

    def raw_control(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def forward(self, x: Tensor) -> Tensor:
        return self.action_high * torch.sin(self.raw_control(x))

    def propagate(self, mu_x: Tensor, sigma_x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        m_raw = mu_x @ self.weight + self.bias
        s_raw = self.weight.t() @ sigma_x @ self.weight
        c_raw = sigma_x @ self.weight
        return _sine_squash_moments(m_raw, s_raw, c_raw, self.action_high)


class MLPPolicy(nn.Module):
    """Simple MLP policy for particle-based Deep PILCO rollouts."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        action_high: Tensor | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")
        if action_high is None:
            action_high = torch.ones(action_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("action_high", torch.as_tensor(action_high).flatten())

    def raw_control(self, x: Tensor) -> Tensor:
        return self.net(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.action_high * torch.tanh(self.raw_control(x))


def build_policy(
    policy_type: str,
    state_dim: int,
    action_dim: int,
    *,
    action_high: Tensor,
    n_basis: int = 50,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
) -> nn.Module:
    """Factory kept small and explicit on purpose."""
    if policy_type == "rbf":
        return RBFPolicy(
            state_dim,
            action_dim,
            n_basis=n_basis,
            action_high=action_high,
        )
    if policy_type == "linear_sine":
        return LinearSinePolicy(state_dim, action_dim, action_high=action_high)
    if policy_type == "mlp":
        return MLPPolicy(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            action_high=action_high,
        )
    raise ValueError(f"Unknown policy_type={policy_type!r}.")
