"""Tabular reinforcement learning agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_from_scratch.core.base import BaseAgent


class QLearningAgent(BaseAgent):
    """Vanilla tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self,
        state_shape: tuple[int, ...] | None = None,
        action_count: int | None = None,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        rng: np.random.Generator | None = None,
        q_table: torch.Tensor | None = None,
    ) -> None:

        if state_shape is None:
            raise ValueError("state_shape is required.")
        if action_count is None or action_count <= 0:
            raise ValueError("action_count must be positive.")
        if any(size <= 0 for size in state_shape):
            raise ValueError("state_shape entries must be positive.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")

        self.state_shape = state_shape
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = rng or np.random.default_rng()

        expected_shape = (*state_shape, action_count)
        if q_table is None:
            self.q_table = torch.rand(expected_shape, dtype=torch.float32) * 0.01
        else:
            if q_table.shape != torch.Size(expected_shape):
                raise ValueError(
                    f"q_table shape must be {expected_shape}, got {tuple(q_table.shape)}."
                )
            self.q_table = q_table.to(dtype=torch.float32).clone()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> int:
        """Choose an action given a discretized observation tuple."""
        if deterministic:
            return self.select_action_greedy(observation)
        return self.select_action_epsilon_greedy(observation)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Run one Q-learning update. Expects state, action, reward, next_state, done."""
        error = self.learn(
            state=kwargs["state"],
            action=kwargs["action"],
            reward=kwargs["reward"],
            next_state=kwargs["next_state"],
            done=kwargs["done"],
        )
        return {"td_error": abs(error)}

    def episode_ended(self) -> None:
        """Decay epsilon at the end of each episode."""
        self.decay_epsilon(self.epsilon_decay, self.min_epsilon)

    # ------------------------------------------------------------------
    # Q-learning methods
    # ------------------------------------------------------------------

    def select_action_random(self) -> int:
        return int(self.rng.integers(self.action_count))

    def select_action_greedy(self, state: tuple[int, ...]) -> int:
        values = self.q_table[state]
        max_val = torch.max(values)
        best_actions = torch.nonzero(values == max_val, as_tuple=False).flatten()
        idx = int(self.rng.integers(len(best_actions)))
        return int(best_actions[idx].item())

    def select_action_epsilon_greedy(self, state: tuple[int, ...]) -> int:
        if float(self.rng.random()) < self.epsilon:
            return self.select_action_random()
        return self.select_action_greedy(state)

    def learn(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> float:
        state_action = (*state, action)
        bootstrap = 0.0 if done else float(torch.max(self.q_table[next_state]).item())
        target = reward + self.gamma * bootstrap
        error = target - float(self.q_table[state_action].item())
        self.q_table[state_action] += self.alpha * error
        return float(error)

    def decay_epsilon(self, decay: float, minimum: float) -> float:
        self.epsilon = max(minimum, self.epsilon * decay)
        return self.epsilon

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_table, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        action_count: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> QLearningAgent:
        q_table = torch.load(Path(path), weights_only=True)
        return cls(
            state_shape=tuple(q_table.shape[:-1]),
            action_count=action_count,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            rng=rng,
            q_table=q_table,
        )


class SarsaAgent(BaseAgent):
    """Tabular SARSA (on-policy) with epsilon-greedy exploration.

    Unlike Q-learning, SARSA uses the action actually taken in the next
    state rather than the greedy maximum, making it an on-policy algorithm.
    """

    def __init__(
        self,
        state_shape: tuple[int, ...] | None = None,
        action_count: int | None = None,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        rng: np.random.Generator | None = None,
        q_table: torch.Tensor | None = None,
    ) -> None:
        if state_shape is None:
            raise ValueError("state_shape is required.")
        if action_count is None or action_count <= 0:
            raise ValueError("action_count must be positive.")
        if any(size <= 0 for size in state_shape):
            raise ValueError("state_shape entries must be positive.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")

        self.state_shape = state_shape
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = rng or np.random.default_rng()

        expected_shape = (*state_shape, action_count)
        if q_table is None:
            self.q_table = torch.rand(expected_shape, dtype=torch.float32) * 0.01
        else:
            if q_table.shape != torch.Size(expected_shape):
                raise ValueError(
                    f"q_table shape must be {expected_shape}, got {tuple(q_table.shape)}."
                )
            self.q_table = q_table.to(dtype=torch.float32).clone()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> int:
        """Choose an action given a discretized observation tuple."""
        if deterministic:
            return self.select_action_greedy(observation)
        return self.select_action_epsilon_greedy(observation)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Run one SARSA update.

        Expects state, action, reward, next_state, next_action, done.
        """
        error = self.learn(
            state=kwargs["state"],
            action=kwargs["action"],
            reward=kwargs["reward"],
            next_state=kwargs["next_state"],
            next_action=kwargs["next_action"],
            done=kwargs["done"],
        )
        return {"td_error": abs(error)}

    def episode_ended(self) -> None:
        """Decay epsilon at the end of each episode."""
        self.decay_epsilon(self.epsilon_decay, self.min_epsilon)

    # ------------------------------------------------------------------
    # SARSA methods
    # ------------------------------------------------------------------

    def select_action_random(self) -> int:
        return int(self.rng.integers(self.action_count))

    def select_action_greedy(self, state: tuple[int, ...]) -> int:
        values = self.q_table[state]
        max_val = torch.max(values)
        best_actions = torch.nonzero(values == max_val, as_tuple=False).flatten()
        idx = int(self.rng.integers(len(best_actions)))
        return int(best_actions[idx].item())

    def select_action_epsilon_greedy(self, state: tuple[int, ...]) -> int:
        if float(self.rng.random()) < self.epsilon:
            return self.select_action_random()
        return self.select_action_greedy(state)

    def learn(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        next_action: int,
        done: bool,
    ) -> float:
        """On-policy SARSA update: uses Q[s', a'] instead of max Q[s', :]."""
        state_action = (*state, action)
        bootstrap = (
            0.0 if done else float(self.q_table[(*next_state, next_action)].item())
        )
        target = reward + self.gamma * bootstrap
        error = target - float(self.q_table[state_action].item())
        self.q_table[state_action] += self.alpha * error
        return float(error)

    def decay_epsilon(self, decay: float, minimum: float) -> float:
        self.epsilon = max(minimum, self.epsilon * decay)
        return self.epsilon

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_table, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        action_count: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> SarsaAgent:
        """Load a saved SARSA agent from a Q-table checkpoint."""
        q_table = torch.load(Path(path), weights_only=True)
        return cls(
            state_shape=tuple(q_table.shape[:-1]),
            action_count=action_count,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            rng=rng,
            q_table=q_table,
        )
