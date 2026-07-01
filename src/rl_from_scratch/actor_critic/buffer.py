"""On-policy rollout buffer for A2C.

Unlike the off-policy replay buffer of DQN, the RolloutBuffer is pre-allocated
for exactly n_steps transitions and is cleared after each update.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-capacity on-policy buffer for collecting rollouts.

    Stores transitions (obs, action, reward, done, log_prob, value) as
    pre-allocated NumPy arrays, then returns them as PyTorch tensors when
    computing returns/advantages.

    Parameters
    ----------
    n_steps:
        Number of time steps to collect before each update.
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    """

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Pre-allocated arrays to avoid dynamic allocations
        self._obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self._actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self._rewards = np.zeros(n_steps, dtype=np.float32)
        self._dones = np.zeros(n_steps, dtype=np.float32)
        self._log_probs = np.zeros(n_steps, dtype=np.float32)
        self._values = np.zeros(n_steps, dtype=np.float32)

        self._ptr = 0

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Record a transition at the current pointer.

        Parameters
        ----------
        obs:
            Observation of the current state.
        action:
            Continuous action taken.
        reward:
            Reward received.
        done:
            End-of-episode flag (terminated or truncated).
        log_prob:
            Log-probability of the action under the current policy.
        value:
            Critic's V(s) estimate for the current state.
        """
        assert self._ptr < self.n_steps, "Buffer full — call reset() before push()."
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._log_probs[self._ptr] = log_prob
        self._values[self._ptr] = value
        self._ptr += 1

    def is_full(self) -> bool:
        """Return True if the buffer holds exactly n_steps transitions."""
        return self._ptr == self.n_steps

    def compute_returns(
        self, next_value: float, gamma: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute N-step returns and advantages (without GAE).

        Returns: R_t = r_t + γ * R_{t+1} * (1 - done_t)
        Advantages: A_t = R_t - V(s_t)

        Parameters
        ----------
        next_value:
            V(s_{T+1}) estimate for the state following the end of the rollout
            (0.0 if the episode has terminated).
        gamma:
            Discount factor.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(returns, advantages)`` each of shape ``(num_collected_steps,)``.
        """
        length = self._ptr
        returns = np.zeros(length, dtype=np.float32)
        G = next_value
        for t in reversed(range(length)):
            G = self._rewards[t] + gamma * G * (1.0 - self._dones[t])
            returns[t] = G

        returns_t = torch.as_tensor(returns)
        values_t = torch.as_tensor(self._values[:length].copy())
        advantages_t = returns_t - values_t
        return returns_t, advantages_t

    def compute_gae(
        self, next_value: float, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages via GAE (Generalized Advantage Estimation).

        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = δ_t + γ * λ * A_{t+1} * (1 - done_t)
        R_t = A_t + V(s_t)  (returns for the value loss)

        Parameters
        ----------
        next_value:
            V(s_{T+1}) estimate after the end of the rollout.
        gamma:
            Discount factor.
        gae_lambda:
            Exponential smoothing parameter λ (0 = TD(0), 1 = Monte Carlo).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(returns, advantages)`` each of shape ``(num_collected_steps,)``.
        """
        length = self._ptr
        advantages = np.zeros(length, dtype=np.float32)
        gae = 0.0
        next_val = next_value

        for t in reversed(range(length)):
            delta = (
                self._rewards[t]
                + gamma * next_val * (1.0 - self._dones[t])
                - self._values[t]
            )
            gae = delta + gamma * gae_lambda * gae * (1.0 - self._dones[t])
            advantages[t] = gae
            next_val = self._values[t]

        advantages_t = torch.as_tensor(advantages)
        values_t = torch.as_tensor(self._values[:length].copy())
        returns_t = advantages_t + values_t
        return returns_t, advantages_t

    def get_batch(
        self, device: "torch.device | None" = None
    ) -> dict[str, torch.Tensor]:
        """Return all stored transitions as tensors.

        Parameters
        ----------
        device:
            Optional target device; if provided, all tensors are moved to
            that device.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``obs``, ``actions``, ``log_probs``, ``values``.
            Shapes: ``(n_steps, obs_dim)``, ``(n_steps, action_dim)``,
            ``(n_steps,)``, ``(n_steps,)``.
        """
        batch = {
            "obs": torch.as_tensor(self._obs[: self._ptr].copy()),
            "actions": torch.as_tensor(self._actions[: self._ptr].copy()),
            "log_probs": torch.as_tensor(self._log_probs[: self._ptr].copy()),
            "values": torch.as_tensor(self._values[: self._ptr].copy()),
        }
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def reset(self) -> None:
        """Reset the pointer to zero to reuse the buffer."""
        self._ptr = 0
