"""Neural networks for REINFORCE and REINFORCE with baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Fully connected policy network.

    Maps observations to action logits for a categorical distribution.
    The softmax is *not* applied here — ``Categorical(logits=...)``
    handles it internally.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    n_actions:
        Number of discrete actions.
    hidden_dim:
        Width of each hidden layer.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the action logits for the given observations.

        Parameters
        ----------
        x:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_actions)``.
        """
        return self.net(x)


class ValueNetwork(nn.Module):
    """Fully connected value network for the baseline.

    Maps observations to a scalar estimate V(s).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    hidden_dim:
        Width of each hidden layer.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute V(s) for the given observations.

        Parameters
        ----------
        x:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Values of shape ``(batch,)`` after removing the last dimension.
        """
        return self.net(x).squeeze(-1)
