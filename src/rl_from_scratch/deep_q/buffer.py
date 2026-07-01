"""Replay buffers and n-step transition helpers for Deep Q agents."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer backed by a deque.

    Stores transitions as ``(state, action, reward, next_state, done)``
    tuples and returns batches as GPU/CPU-ready tensors.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[tuple[Any, int, float, Any, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append((
            np.asarray(state, dtype=np.float32),
            action,
            reward,
            np.asarray(next_state, dtype=np.float32),
            done,
        ))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch and return tensors.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            ``(states, actions, rewards, next_states, dones)`` with shapes
            ``(B, obs_dim)``, ``(B,)``, ``(B,)``, ``(B, obs_dim)``, ``(B,)``.
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *(self._buffer[i] for i in indices)
        )
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class PrioritizedReplayBuffer:
    """Simple proportional prioritized replay buffer."""

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 100_000,
        eps: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = max(1, beta_annealing_steps)
        self.eps = eps
        self._storage: list[tuple[Any, int, float, Any, bool]] = []
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._max_priority = 1.0
        self._beta_increment = (1.0 - beta) / self.beta_annealing_steps

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self._storage) < batch_size:
            raise ValueError("Not enough samples in buffer.")

        current_size = len(self._storage)
        priorities = self._priorities[:current_size]
        scaled = priorities**self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(current_size, size=batch_size, replace=False, p=probs)

        states, actions, rewards, next_states, dones = zip(*(self._storage[i] for i in indices))

        weights = (current_size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self._beta_increment)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, indices: Any, priorities: Any) -> None:
        indices_np = np.asarray(indices, dtype=np.int64)
        priorities_np = np.asarray(priorities, dtype=np.float32)
        priorities_np = np.maximum(priorities_np, self.eps)
        self._priorities[indices_np] = priorities_np
        self._max_priority = max(self._max_priority, float(priorities_np.max(initial=self._max_priority)))

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable replay state for checkpoint resumes."""
        return {
            "storage": self._storage,
            "priorities": self._priorities,
            "pos": self._pos,
            "max_priority": self._max_priority,
            "beta": self.beta,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay state saved by :meth:`state_dict`."""
        self._storage = list(state.get("storage", []))
        priorities = np.asarray(state.get("priorities", self._priorities), dtype=np.float32)
        self._priorities.fill(0.0)
        self._priorities[: len(priorities)] = priorities[: self.capacity]
        self._pos = int(state.get("pos", len(self._storage) % self.capacity))
        self._max_priority = float(state.get("max_priority", 1.0))
        self.beta = float(state.get("beta", self.beta))

    def __len__(self) -> int:
        return len(self._storage)


class NStepTransitionAccumulator:
    """Accumulate n-step returns and flush trailing transitions on episode end."""

    def __init__(self, n_steps: int, gamma: float) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        self.n_steps = n_steps
        self.gamma = gamma
        self._buffer: deque[tuple[Any, int, float, Any, bool]] = deque()

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> list[tuple[Any, int, float, Any, bool]]:
        self._buffer.append((state, int(action), float(reward), next_state, bool(done)))
        emitted: list[tuple[Any, int, float, Any, bool]] = []

        if done:
            while self._buffer:
                emitted.append(self._aggregate_transition(len(self._buffer)))
                self._buffer.popleft()
            return emitted

        if len(self._buffer) >= self.n_steps:
            emitted.append(self._aggregate_transition(self.n_steps))
            self._buffer.popleft()

        return emitted

    def flush(self) -> list[tuple[Any, int, float, Any, bool]]:
        """Emit all pending partial n-step transitions and clear the accumulator."""
        emitted: list[tuple[Any, int, float, Any, bool]] = []
        while self._buffer:
            emitted.append(self._aggregate_transition(len(self._buffer)))
            self._buffer.popleft()
        return emitted

    def state_dict(self) -> dict[str, Any]:
        """Return pending n-step transitions for checkpoint resumes."""
        return {"buffer": list(self._buffer)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore pending n-step transitions saved by :meth:`state_dict`."""
        self._buffer = deque(state.get("buffer", []))

    def _aggregate_transition(self, length: int) -> tuple[Any, int, float, Any, bool]:
        state, action, _, _, _ = self._buffer[0]
        reward = 0.0
        next_state = self._buffer[length - 1][3]
        done = self._buffer[length - 1][4]
        for idx in range(length):
            reward += (self.gamma**idx) * self._buffer[idx][2]
        return state, action, reward, next_state, done
