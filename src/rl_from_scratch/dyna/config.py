"""Configuration objects for the autonomous dyna package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("dyna_q")
@dataclass
class DynaQConfig(BaseConfig):
    """Configuration for tabular Dyna-Q on discretized CartPole."""

    approach: str = "dyna_q"
    env_id: str = "CartPole-v1"
    episodes: int = 300  # type: ignore[assignment]
    total_timesteps: int = 50_000
    solved_reward: float | None = 195.0

    bins: tuple[int, int, int, int] = (12, 12, 12, 12)
    cart_velocity_min: float = -3.0
    cart_velocity_max: float = 3.0
    pole_angular_velocity_min: float = -10.0
    pole_angular_velocity_max: float = 10.0
    alpha: float = 0.2
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.02
    planning_steps: int = 10
    random_episodes: int = 0
    checkpoint_every: int = 25

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.env_id != "CartPole-v1":
            raise ValueError("Dyna-Q currently supports only CartPole-v1.")
        if len(self.bins) != 4 or any(count <= 1 for count in self.bins):
            raise ValueError("bins must contain four integers greater than 1.")
        if self.cart_velocity_min >= self.cart_velocity_max:
            raise ValueError("cart velocity bounds are invalid.")
        if self.pole_angular_velocity_min >= self.pole_angular_velocity_max:
            raise ValueError("pole angular velocity bounds are invalid.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 <= self.min_epsilon <= self.epsilon:
            raise ValueError("min_epsilon must be in [0, epsilon].")
        if self.planning_steps < 0:
            raise ValueError("planning_steps must be non-negative.")
        if self.random_episodes < 0:
            raise ValueError("random_episodes must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bins"] = list(self.bins)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DynaQConfig:
        return strict_dataclass_from_dict(
            cls,
            payload,
            converters={"bins": lambda values: tuple(int(value) for value in values)},
        )


@register_config("dyna_q_plus")
@dataclass
class DynaQPlusConfig(DynaQConfig):
    """Configuration for Dyna-Q+ with planning-time exploration bonus."""

    approach: str = "dyna_q_plus"
    kappa: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DynaQPlusConfig:
        return strict_dataclass_from_dict(
            cls,
            payload,
            converters={"bins": lambda values: tuple(int(value) for value in values)},
        )


@register_config("deep_dyna")
@dataclass
class DeepDynaConfig(BaseConfig):
    """Configuration for a simple Deep Dyna variant on CartPole."""

    approach: str = "deep_dyna"
    env_id: str = "CartPole-v1"
    episodes: int = 200  # type: ignore[assignment]
    total_timesteps: int = 100_000
    solved_reward: float | None = 195.0

    hidden_dim: int = 64
    lr: float = 1e-3
    model_lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    buffer_capacity: int = 20_000
    batch_size: int = 64
    target_update_freq: int = 100
    model_train_steps: int = 1
    imagined_updates: int = 4
    start_learning_after: int = 1_000
    checkpoint_every: int = 25

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.env_id != "CartPole-v1":
            raise ValueError("Deep Dyna currently supports only CartPole-v1.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.model_lr <= 0.0:
            raise ValueError("model_lr must be positive.")
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
        if self.model_train_steps < 0:
            raise ValueError("model_train_steps must be non-negative.")
        if self.imagined_updates < 0:
            raise ValueError("imagined_updates must be non-negative.")
        if self.start_learning_after < 0:
            raise ValueError("start_learning_after must be non-negative.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DeepDynaConfig:
        return strict_dataclass_from_dict(cls, payload)
