"""Configuration for DQN-family experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("dqn")
@dataclass
class DQNConfig(BaseConfig):
    """Configuration for DQN training on continuous-observation environments."""

    # Override BaseConfig defaults
    approach: str = "dqn"
    gamma: float = 0.99
    episodes: int = 300  # type: ignore[assignment]

    # DQN-specific fields
    hidden_dim: int = 64
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 <= self.min_epsilon <= self.epsilon:
            raise ValueError("min_epsilon must be in [0, epsilon].")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DQNConfig:
        """Construct from a plain dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)


@register_config("double_dqn")
@dataclass
class DoubleDQNConfig(BaseConfig):
    """Configuration for Double DQN training."""

    # Override BaseConfig defaults
    approach: str = "double_dqn"
    gamma: float = 0.99
    episodes: int = 300  # type: ignore[assignment]

    # DQN-specific fields (same as DQNConfig)
    hidden_dim: int = 64
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 <= self.min_epsilon <= self.epsilon:
            raise ValueError("min_epsilon must be in [0, epsilon].")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DoubleDQNConfig:
        """Construct from a plain dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)


@register_config("rainbow_dqn")
@dataclass
class RainbowDQNConfig(BaseConfig):
    """Configuration for Rainbow DQN training."""

    approach: str = "rainbow_dqn"
    env_id: str = "CartPole-v1"
    gamma: float = 0.99
    episodes: int = 300  # type: ignore[assignment]

    hidden_dim: int = 64
    lr: float = 1e-3
    buffer_capacity: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 100
    n_steps: int = 3
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    noisy_std_init: float = 0.5
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_steps: int = 100_000
    priority_eps: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.target_update_freq <= 0:
            raise ValueError("target_update_freq must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.n_atoms <= 1:
            raise ValueError("n_atoms must be greater than 1.")
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be smaller than v_max.")
        if self.noisy_std_init <= 0:
            raise ValueError("noisy_std_init must be positive.")
        if not 0.0 <= self.priority_alpha <= 1.0:
            raise ValueError("priority_alpha must be in [0, 1].")
        if not 0.0 <= self.priority_beta <= 1.0:
            raise ValueError("priority_beta must be in [0, 1].")
        if self.priority_beta_steps <= 0:
            raise ValueError("priority_beta_steps must be positive.")
        if self.priority_eps <= 0:
            raise ValueError("priority_eps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RainbowDQNConfig:
        return strict_dataclass_from_dict(cls, payload)
