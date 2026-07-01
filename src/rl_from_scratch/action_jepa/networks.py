"""Neural building blocks for Action-JEPA."""

from __future__ import annotations

import torch
from torch import nn

from rl_from_scratch.core.utils import soft_update


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(max(1, n_layers)):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Observation encoder used by both the online and target towers."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.backbone = _mlp(obs_dim, hidden_dim, latent_dim, n_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.backbone(obs))


class Predictor(nn.Module):
    """Action-conditioned latent predictor."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        delta: bool = True,
    ) -> None:
        super().__init__()
        self.delta = bool(delta)
        self.net = _mlp(latent_dim + action_dim, hidden_dim, latent_dim, n_layers=2)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        update = self.net(torch.cat([latent, action], dim=-1))
        return latent + update if self.delta else update


class MaskedContextPredictor(nn.Module):
    """Predict a target latent from context, partial target view, and mask.

    This is the Phase-A JEPA head used by the stage-wise regime: the online
    encoder sees the current observation and a masked/noisy view of the next
    observation, while the EMA teacher encodes the full next observation.
    """

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(2 * latent_dim + obs_dim, hidden_dim, latent_dim, n_layers=2)

    def forward(
        self,
        context_latent: torch.Tensor,
        partial_latent: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([context_latent, partial_latent, visible_mask], dim=-1)
        return self.net(inputs)


class RewardHead(nn.Module):
    """Predict rewards from the current latent state and action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden_dim, 1, n_layers=2)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1)).squeeze(-1)


class ContinuationHead(nn.Module):
    """Predict the probability that the episode continues."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(latent_dim + action_dim, hidden_dim, 1, n_layers=2)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1)).squeeze(-1)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Update ``target`` toward ``online`` with an exponential moving average.

    Here *tau* is the weight kept on the slow target (e.g. ``0.99``), so the
    update is ``target ← tau·target + (1 - tau)·online``. This delegates to the
    shared :func:`rl_from_scratch.core.utils.soft_update`, whose convention puts
    weight ``tau`` on the *source* (the DDPG/SAC convention) — hence ``1 - tau``.
    """
    soft_update(target, online, 1.0 - tau)


def variance_loss(latents: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """VICReg variance term: each latent dimension should keep some spread."""
    centered = latents - latents.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(target_std - std).mean()


def covariance_loss(latents: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: discourage redundant latent dimensions."""
    centered = latents - latents.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    # VICReg normalization: sum of the squared off-diagonal terms, divided by d
    # (not .mean() over d*d, which underweights the term by a factor of 1/d).
    return off_diag.pow(2).sum() / latents.shape[1]


def latent_collapse_metric(latents: torch.Tensor) -> float:
    """Return the average per-dimension latent standard deviation."""
    centered = latents - latents.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    return float(std.mean().detach().cpu().item())
