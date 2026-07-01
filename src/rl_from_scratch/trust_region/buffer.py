"""Local on-policy rollout buffer for trust-region methods."""

from __future__ import annotations

import numpy as np
import torch


class TrustRegionRolloutBuffer:
    """Fixed-size rollout buffer with local GAE computation."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim

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
        assert self._ptr < self.n_steps, "Buffer full. Call reset() before pushing again."
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._log_probs[self._ptr] = log_prob
        self._values[self._ptr] = value
        self._ptr += 1

    def is_full(self) -> bool:
        return self._ptr == self.n_steps

    def compute_gae(
        self,
        next_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = self._ptr
        advantages = np.zeros(length, dtype=np.float32)
        gae = 0.0
        bootstrap_value = next_value

        for step in reversed(range(length)):
            delta = (
                self._rewards[step]
                + gamma * bootstrap_value * (1.0 - self._dones[step])
                - self._values[step]
            )
            gae = delta + gamma * gae_lambda * gae * (1.0 - self._dones[step])
            advantages[step] = gae
            bootstrap_value = self._values[step]

        advantages_t = torch.as_tensor(advantages.copy())
        values_t = torch.as_tensor(self._values[:length].copy())
        returns_t = advantages_t + values_t
        return returns_t, advantages_t

    def get_batch(
        self,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = {
            "obs": torch.as_tensor(self._obs[: self._ptr].copy()),
            "actions": torch.as_tensor(self._actions[: self._ptr].copy()),
            "log_probs": torch.as_tensor(self._log_probs[: self._ptr].copy()),
            "values": torch.as_tensor(self._values[: self._ptr].copy()),
        }
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch

    def reset(self) -> None:
        self._ptr = 0
