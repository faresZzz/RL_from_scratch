"""Episode-aware sequence replay for Action-JEPA."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SequenceBuffer:
    """Store full episodes and sample contiguous latent-prediction windows."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self._episodes: list[dict[str, np.ndarray]] = []
        self._current: list[tuple[Any, Any, float, Any, bool]] = []
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._current.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
            )
        )
        if done:
            self.flush_current()

    def flush_current(self) -> None:
        if not self._current:
            return
        obs, action, reward, next_obs, done = zip(*self._current)
        episode = {
            "obs": np.stack(obs).astype(np.float32),
            "action": np.stack(action).astype(np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "next_obs": np.stack(next_obs).astype(np.float32),
            "done": np.asarray(done, dtype=np.float32),
        }
        self._episodes.append(episode)
        self._current = []
        while self._total_transitions() > self.capacity and len(self._episodes) > 1:
            self._episodes.pop(0)

    def _total_transitions(self) -> int:
        return sum(episode["obs"].shape[0] for episode in self._episodes)

    def sample(self, batch_size: int, rollout_len: int) -> dict[str, torch.Tensor]:
        window_len = rollout_len + 1
        eligible = [episode for episode in self._episodes if episode["obs"].shape[0] >= window_len]
        if not eligible:
            raise RuntimeError(
                f"No episode is long enough for rollout_len={rollout_len}. "
                f"Episode lengths: {[episode['obs'].shape[0] for episode in self._episodes]}"
            )

        obs_batch: list[np.ndarray] = []
        next_obs_batch: list[np.ndarray] = []
        action_batch: list[np.ndarray] = []
        reward_batch: list[np.ndarray] = []
        done_batch: list[np.ndarray] = []

        for _ in range(batch_size):
            episode = eligible[int(self._rng.integers(len(eligible)))]
            length = episode["obs"].shape[0]
            start = int(self._rng.integers(0, length - window_len + 1))
            stop = start + window_len
            obs_window = episode["obs"][start:stop]
            next_obs_window = np.concatenate(
                [obs_window[1:], episode["next_obs"][stop - 1:stop]],
                axis=0,
            )
            obs_batch.append(obs_window)
            next_obs_batch.append(next_obs_window)
            action_batch.append(episode["action"][start:stop - 1])
            reward_batch.append(episode["reward"][start:stop - 1])
            done_batch.append(episode["done"][start:stop - 1])

        return {
            "obs": torch.tensor(np.stack(obs_batch), dtype=torch.float32),
            "next_obs": torch.tensor(np.stack(next_obs_batch), dtype=torch.float32),
            "action": torch.tensor(np.stack(action_batch), dtype=torch.float32),
            "reward": torch.tensor(np.stack(reward_batch), dtype=torch.float32),
            "done": torch.tensor(np.stack(done_batch), dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self._total_transitions() + len(self._current)
