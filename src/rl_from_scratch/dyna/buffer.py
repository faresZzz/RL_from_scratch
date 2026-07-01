"""Local replay buffer for Deep Dyna."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    """Simple FIFO replay buffer with tensor sampling."""

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.capacity = capacity
        self.rng = rng or np.random.default_rng()
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = rng or self.rng
        indices = generator.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self._buffer[i] for i in indices))
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(np.asarray(actions), dtype=torch.long),
            torch.tensor(np.asarray(rewards), dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(np.asarray(dones), dtype=torch.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "storage": list(self._buffer),
            "capacity": self.capacity,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        storage = payload.get("storage", [])
        self._buffer = deque(storage, maxlen=self.capacity)
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def __len__(self) -> int:
        return len(self._buffer)
