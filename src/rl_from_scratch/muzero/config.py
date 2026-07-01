"""Configuration for the autonomous MuZero package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("muzero")
@dataclass
class MuZeroConfig(BaseConfig):
    """Pedagogical MuZero configuration for discrete environments."""

    approach: str = "muzero"
    env_id: str = "CartPole-v1"
    discount: float = 0.997

    training_steps: int = 20
    selfplay_episodes_per_iteration: int = 1
    updates_per_iteration: int = 8
    replay_capacity: int = 500
    num_warmup_games: int = 4

    encoding_dim: int = 8
    hidden_dim: int = 32
    support_size: int = 10

    num_simulations: int = 25
    pb_c_base: float = 19_652.0
    pb_c_init: float = 1.25
    dirichlet_alpha: float = 0.25
    exploration_fraction: float = 0.25
    root_temperature: float = 1.0
    root_temperature_drop_episode: int | None = None

    num_unroll_steps: int = 5
    td_steps: int = 10
    batch_size: int = 64
    lr: float = 0.02
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    value_loss_weight: float = 0.25
    two_player: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_steps <= 0:
            raise ValueError("training_steps must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if self.selfplay_episodes_per_iteration <= 0:
            raise ValueError("selfplay_episodes_per_iteration must be positive.")
        if self.updates_per_iteration <= 0:
            raise ValueError("updates_per_iteration must be positive.")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive.")
        if self.num_warmup_games < 0:
            raise ValueError("num_warmup_games must be non-negative.")
        if self.encoding_dim <= 0:
            raise ValueError("encoding_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.support_size <= 0:
            raise ValueError("support_size must be positive.")
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")
        if self.pb_c_base <= 0.0:
            raise ValueError("pb_c_base must be positive.")
        if self.pb_c_init <= 0.0:
            raise ValueError("pb_c_init must be positive.")
        if not 0.0 <= self.exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in [0, 1].")
        if self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive.")
        if self.root_temperature <= 0.0:
            raise ValueError("root_temperature must be positive.")
        if self.root_temperature_drop_episode is not None and self.root_temperature_drop_episode < 0:
            raise ValueError("root_temperature_drop_episode must be non-negative.")
        if self.num_unroll_steps < 1:
            raise ValueError("num_unroll_steps must be >= 1.")
        if self.td_steps < 1:
            raise ValueError("td_steps must be >= 1.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive.")
        if self.value_loss_weight <= 0.0:
            raise ValueError("value_loss_weight must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MuZeroConfig":
        return strict_dataclass_from_dict(cls, payload)
