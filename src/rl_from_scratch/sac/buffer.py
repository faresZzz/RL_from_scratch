"""SAC replay buffer for continuous actions.

The buffer is part of the SAC chapter so that the algorithm stays self-contained,
even though its structure is close to DDPG/TD3.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class ContinuousReplayBuffer:
    """Fixed-capacity FIFO buffer for continuous transitions."""

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[
            tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: np.ndarray,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a random batch without replacement and return tensors."""
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *(self._buffer[i] for i in indices)
        )
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

