"""Neural components and scalar/support helpers for MuZero."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MUZERO_EPSILON = 1e-3


def scalar_transform(values: torch.Tensor, epsilon: float = MUZERO_EPSILON) -> torch.Tensor:
    abs_values = values.abs()
    return values.sign() * (torch.sqrt(abs_values + 1.0) - 1.0) + epsilon * values


def inverse_scalar_transform(values: torch.Tensor, epsilon: float = MUZERO_EPSILON) -> torch.Tensor:
    abs_values = values.abs()
    inside = torch.sqrt(1.0 + 4.0 * epsilon * (abs_values + 1.0 + epsilon))
    base = ((inside - 1.0) / (2.0 * epsilon)) ** 2 - 1.0
    return values.sign() * base


def scalar_to_support(
    values: torch.Tensor,
    support_size: int,
    *,
    apply_transform: bool = False,
) -> torch.Tensor:
    transformed = scalar_transform(values) if apply_transform else values
    clipped = transformed.clamp(-support_size, support_size)
    floor = clipped.floor()
    prob_high = clipped - floor
    prob_low = 1.0 - prob_high
    floor_index = (floor + support_size).long()
    ceil_index = (clipped.ceil() + support_size).long()
    bins = clipped.new_zeros((*clipped.shape, 2 * support_size + 1))
    bins.scatter_add_(-1, floor_index.unsqueeze(-1), prob_low.unsqueeze(-1))
    bins.scatter_add_(-1, ceil_index.unsqueeze(-1), prob_high.unsqueeze(-1))
    return bins


def support_to_scalar(
    support: torch.Tensor,
    support_size: int,
    *,
    apply_inverse: bool = False,
) -> torch.Tensor:
    support_range = torch.arange(
        -support_size,
        support_size + 1,
        device=support.device,
        dtype=support.dtype,
    )
    if support.dim() == 1:
        support = support.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    expectations = (support * support_range).sum(dim=-1)
    recovered = inverse_scalar_transform(expectations) if apply_inverse else expectations
    return recovered.squeeze(0) if squeeze else recovered


def scale_hidden_01(hidden_state: torch.Tensor) -> torch.Tensor:
    minimum = hidden_state.amin(dim=-1, keepdim=True)
    maximum = hidden_state.amax(dim=-1, keepdim=True)
    span = maximum - minimum
    safe_span = torch.where(span > 1e-8, span, torch.ones_like(span))
    scaled = (hidden_state - minimum) / safe_span
    if hidden_state.shape[-1] == 1:
        return torch.zeros_like(hidden_state)
    return scaled


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Representation(nn.Module):
    """Representation function h: observation -> latent state."""

    def __init__(self, obs_dim: int, hidden_dim: int, encoding_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, encoding_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.net(observation)
        return scale_hidden_01(hidden)


class Dynamics(nn.Module):
    """Dynamics function g: (latent, action) -> (next latent, reward logits)."""

    def __init__(
        self,
        encoding_dim: int,
        action_dim: int,
        hidden_dim: int,
        support_size: int,
    ) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim
        self.action_dim = action_dim
        self.support_size = support_size
        self.transition = _mlp(encoding_dim + action_dim, hidden_dim, encoding_dim)
        self.reward_head = _mlp(encoding_dim, hidden_dim, 2 * support_size + 1)

    def forward(
        self,
        hidden_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_one_hot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        transition_input = torch.cat([scale_hidden_01(hidden_state), action_one_hot], dim=-1)
        next_hidden = scale_hidden_01(self.transition(transition_input))
        reward_logits = self.reward_head(next_hidden)
        return next_hidden, reward_logits


class Prediction(nn.Module):
    """Prediction function f: latent -> (policy logits, value logits)."""

    def __init__(
        self,
        encoding_dim: int,
        hidden_dim: int,
        action_dim: int,
        support_size: int,
    ) -> None:
        super().__init__()
        self.policy_head = _mlp(encoding_dim, hidden_dim, action_dim)
        self.value_head = _mlp(encoding_dim, hidden_dim, 2 * support_size + 1)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = scale_hidden_01(hidden_state)
        return self.policy_head(hidden), self.value_head(hidden)
