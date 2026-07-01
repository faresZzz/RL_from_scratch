"""Configuration dataclasses for autonomous trust-region methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@dataclass
class TrustRegionConfig(BaseConfig):
    """Shared configuration surface for TRPO and PPO."""

    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    gamma: float = 0.99
    checkpoint_every: int = 50_000

    hidden_dim: int = 256
    lr: float = 3e-4
    n_steps: int = 2048
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95

    normalize_observations: bool = False
    obs_norm_epsilon: float = 1e-8
    obs_norm_clip: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative.")
        if self.value_coef < 0:
            raise ValueError("value_coef must be non-negative.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrustRegionConfig:
        return strict_dataclass_from_dict(cls, payload)


@register_config("trpo")
@dataclass
class TRPOConfig(TrustRegionConfig):
    """TRPO config with natural-gradient and line-search controls."""

    approach: str = "trpo"
    max_kl: float = 0.01
    cg_iters: int = 10
    cg_damping: float = 0.1
    backtrack_iters: int = 10
    backtrack_coeff: float = 0.8
    value_train_iters: int = 80
    value_lr: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_kl <= 0:
            raise ValueError("max_kl must be positive.")
        if self.cg_iters <= 0:
            raise ValueError("cg_iters must be positive.")
        if self.cg_damping < 0:
            raise ValueError("cg_damping must be non-negative.")
        if self.backtrack_iters <= 0:
            raise ValueError("backtrack_iters must be positive.")
        if not 0.0 < self.backtrack_coeff < 1.0:
            raise ValueError("backtrack_coeff must be in (0, 1).")
        if self.value_train_iters <= 0:
            raise ValueError("value_train_iters must be positive.")
        if self.value_lr <= 0:
            raise ValueError("value_lr must be positive.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TRPOConfig:
        return strict_dataclass_from_dict(cls, payload)


@register_config("ppo")
@dataclass
class PPOConfig(TrustRegionConfig):
    """PPO config with clip-ratio and minibatch SGD controls."""

    approach: str = "ppo"
    clip_ratio: float = 0.2
    n_epochs: int = 10
    batch_size: int = 64
    target_kl: float = 0.01
    entropy_coef: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.target_kl <= 0:
            raise ValueError("target_kl must be positive.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PPOConfig:
        return strict_dataclass_from_dict(cls, payload)
