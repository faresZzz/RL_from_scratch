"""Replay buffer for DDPG and TD3 agents (continuous actions).

Stores transitions (s, a, r, s', done) where actions are float32 numpy
arrays of shape (action_dim,). Returns pytorch tensors ready for gradient
computation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class ContinuousReplayBuffer:
    """Fixed-capacity FIFO replay buffer for continuous actions.

    Stores transitions as tuples and returns GPU/CPU-compatible tensor
    batches for off-policy training.

    Parameters
    ----------
    capacity:
        Maximum number of stored transitions. The oldest ones are
        overwritten in FIFO order once capacity is reached.
    """

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
        """Add a transition to the buffer.

        Parameters
        ----------
        state:
            Current observation.
        action:
            Continuous action taken, stored as float32.
        reward:
            Scalar reward received.
        next_state:
            Next observation.
        done:
            True if the episode ended (termination or truncation).
        """
        self._buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a random batch without replacement and return tensors.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            ``(states, actions, rewards, next_states, dones)`` avec formes
            ``(B, obs_dim)``, ``(B, action_dim)``, ``(B,)``, ``(B, obs_dim)``, ``(B,)``.
        """
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
