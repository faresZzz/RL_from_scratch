"""Configuration for tabular Q-learning experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("q_learning")
@dataclass
class QLearningConfig(BaseConfig):
    """Configuration for tabular Q-learning on discretized CartPole."""

    # Override BaseConfig defaults for Q-learning
    gamma: float = 1.0
    episodes: int = 15_000  # type: ignore[assignment]

    # Q-learning specific fields
    bins: tuple[int, int, int, int] = (30, 30, 30, 30)
    cart_velocity_min: float = -3.0
    cart_velocity_max: float = 3.0
    pole_angular_velocity_min: float = -10.0
    pole_angular_velocity_max: float = 10.0
    alpha: float = 0.1
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.02
    random_episodes: int = 0
    checkpoint_every: int = 50

    def __post_init__(self) -> None:
        if self.min_epsilon > self.epsilon:
            self.min_epsilon = self.epsilon
        if self.env_id != "CartPole-v1":
            raise ValueError("Only CartPole-v1 is supported in this phase.")
        if self.episodes is not None and self.episodes <= 0:
            raise ValueError("episodes must be positive.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if len(self.bins) != 4:
            raise ValueError("bins must contain four integers.")
        if any(bin_count <= 1 for bin_count in self.bins):
            raise ValueError("each bin count must be greater than 1.")
        if self.cart_velocity_min >= self.cart_velocity_max:
            raise ValueError("cart velocity bounds are invalid.")
        if self.pole_angular_velocity_min >= self.pole_angular_velocity_max:
            raise ValueError("pole angular velocity bounds are invalid.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 <= self.min_epsilon <= self.epsilon:
            raise ValueError("min_epsilon must be in [0, epsilon].")
        if self.random_episodes < 0:
            raise ValueError("random_episodes must be non-negative.")
        if self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bins"] = list(self.bins)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QLearningConfig:
        """Construct from a plain dict, handling explicit aliases only."""
        aliases = {
            "num_episodes": "episodes",
            "number_of_epoch": "episodes",
            "total_episodes": "episodes",
            "num_bins": "bins",
            "state_bins": "bins",
        }
        return strict_dataclass_from_dict(
            cls,
            payload,
            aliases=aliases,
            ignored_keys={"seeds"},
            converters={"bins": lambda values: tuple(int(value) for value in values)},
        )


def load_q_learning_config(path: str | Path) -> QLearningConfig:
    """Load a Q-learning config from YAML (convenience wrapper)."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}.")
    return QLearningConfig.from_dict(data)


@register_config("sarsa")
@dataclass
class SarsaConfig(BaseConfig):
    """Configuration for tabular SARSA (on-policy) on discretized CartPole."""

    # Override BaseConfig defaults for SARSA
    gamma: float = 1.0
    episodes: int = 15_000  # type: ignore[assignment]

    # SARSA specific fields
    bins: tuple[int, int, int, int] = (30, 30, 30, 30)
    cart_velocity_min: float = -3.0
    cart_velocity_max: float = 3.0
    pole_angular_velocity_min: float = -10.0
    pole_angular_velocity_max: float = 10.0
    alpha: float = 0.1
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.02
    random_episodes: int = 0
    checkpoint_every: int = 50

    def __post_init__(self) -> None:
        if self.min_epsilon > self.epsilon:
            self.min_epsilon = self.epsilon
        if self.env_id != "CartPole-v1":
            raise ValueError("Only CartPole-v1 is supported in this phase.")
        if self.episodes is not None and self.episodes <= 0:
            raise ValueError("episodes must be positive.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if len(self.bins) != 4:
            raise ValueError("bins must contain four integers.")
        if any(bin_count <= 1 for bin_count in self.bins):
            raise ValueError("each bin count must be greater than 1.")
        if self.cart_velocity_min >= self.cart_velocity_max:
            raise ValueError("cart velocity bounds are invalid.")
        if self.pole_angular_velocity_min >= self.pole_angular_velocity_max:
            raise ValueError("pole angular velocity bounds are invalid.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 <= self.min_epsilon <= self.epsilon:
            raise ValueError("min_epsilon must be in [0, epsilon].")
        if self.random_episodes < 0:
            raise ValueError("random_episodes must be non-negative.")
        if self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bins"] = list(self.bins)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SarsaConfig:
        """Construct from a plain dict, handling explicit aliases only."""
        aliases = {
            "num_episodes": "episodes",
            "number_of_epoch": "episodes",
            "total_episodes": "episodes",
            "num_bins": "bins",
            "state_bins": "bins",
        }
        return strict_dataclass_from_dict(
            cls,
            payload,
            aliases=aliases,
            ignored_keys={"seeds"},
            converters={"bins": lambda values: tuple(int(value) for value in values)},
        )


def load_sarsa_config(path: str | Path) -> SarsaConfig:
    """Load a SARSA config from YAML (convenience wrapper)."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}.")
    return SarsaConfig.from_dict(data)
