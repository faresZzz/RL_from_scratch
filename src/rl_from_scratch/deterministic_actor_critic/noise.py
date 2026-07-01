"""Noise processes for exploration in continuous action spaces.

Two noise types are provided:
- ``GaussianNoise``: i.i.d. Gaussian white noise — simple, effective in practice.
- ``OUNoise``: Ornstein-Uhlenbeck process — temporally correlated noise,
  historically used by the original DDPG (Lillicrap et al., 2015).
"""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    """Independent and identically distributed Gaussian noise.

    Adds N(0, σ²) to each action dimension on each call.
    Recommended by default for TD3 (Fujimoto et al., 2018), which shows that
    Gaussian noise often outperforms OU noise in practice.

    Parameters
    ----------
    action_dim:
        Dimensionality of the action space.
    sigma:
        Standard deviation of the Gaussian noise.
    """

    def __init__(self, action_dim: int, sigma: float = 0.1) -> None:
        self.action_dim = action_dim
        self.sigma = sigma

    def __call__(self) -> np.ndarray:
        """Generate a Gaussian noise vector.

        Returns
        -------
        np.ndarray
            Noise of shape ``(action_dim,)`` in float32.
        """
        return np.random.normal(0.0, self.sigma, size=self.action_dim).astype(np.float32)

    def reset(self) -> None:
        """Reset the noise state (no-op for i.i.d. Gaussian noise)."""


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration.

    Generates correlated noise following the stochastic dynamics:
        x_next = x + θ·(μ - x)·dt + σ·N(0, 1)

    With dt = 1 (discrete), this gives:
        x_next = x + θ·(μ - x) + σ·N(0, 1)

    This process is mean-reverting: it tends toward μ at speed θ,
    while being perturbed by Gaussian noise σ.

    Parameters
    ----------
    action_dim:
        Dimensionality of the action space.
    mu:
        Mean value toward which the process reverts (0 by default).
    theta:
        Mean-reversion speed (0.15 by default — original DDPG).
    sigma:
        Noise amplitude (0.2 by default — original DDPG).
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self._state = np.full(action_dim, mu, dtype=np.float32)

    def __call__(self) -> np.ndarray:
        """Generate the next OU noise vector and update the state.

        Returns
        -------
        np.ndarray
            Noise of shape ``(action_dim,)`` in float32.
        """
        x = self._state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self._state = (x + dx).astype(np.float32)
        return self._state.copy()

    def reset(self) -> None:
        """Reset the internal state to the mean value μ.

        Must be called at the start of each episode to prevent noise from
        one episode from contaminating exploration in the next.
        """
        self._state = np.full(self.action_dim, self.mu, dtype=np.float32)
