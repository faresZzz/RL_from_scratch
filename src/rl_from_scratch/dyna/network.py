"""Local networks for Deep Dyna."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Small MLP that maps observations to action-values."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class NeuralDynamicsModel(nn.Module):
    """Predict next observation, reward, and done logits from state-action pairs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        input_dim = obs_dim + n_actions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_one_hot = F.one_hot(actions.long(), num_classes=self.n_actions).float()
        features = self.encoder(torch.cat([obs, action_one_hot], dim=1))
        delta = self.delta_head(features)
        next_obs = obs + delta  # Residual prediction: the network learns delta, not the full next obs.
        reward = self.reward_head(features).squeeze(1)
        done_logits = self.done_head(features).squeeze(1)
        return next_obs, reward, done_logits
