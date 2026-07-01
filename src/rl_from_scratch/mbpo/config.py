"""Configuration for MBPO (Model-Based Policy Optimization, Janner et al. 2019).

MBPO alternates between:
1. Fitting a probabilistic ensemble that predicts (Δstate, reward).
2. Generating short imagined rollouts from the ensemble starting from real states.
3. Updating a SAC policy on a mix of real and imagined transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("mbpo")
@dataclass
class MbpoConfig(BaseConfig):
    """Configuration for MBPO on HalfCheetah-v5 (or any continuous env)."""

    approach: str = "mbpo"
    env_id: str = "HalfCheetah-v5"
    epochs: int = 20
    steps_per_epoch: int = 1_000
    max_steps_per_episode: int = 1_000

    # ── Probabilistic ensemble ────────────────────────────────────────
    ensemble_size: int = 7
    model_hidden_dim: int = 200
    model_n_layers: int = 4
    model_lr: float = 1e-3
    model_fit_steps: int = 200
    model_batch_size: int = 256
    weight_decay: float = 1e-4

    # ── SAC policy ────────────────────────────────────────────────────
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 5e-3
    sac_hidden_dim: int = 256
    sac_batch_size: int = 256
    alpha: float = 0.2
    auto_tune_alpha: bool = True
    target_entropy: float | None = None

    # ── MBPO rollout schedule ─────────────────────────────────────────
    rollout_length: int = 1
    rollout_batch_size: int = 400
    rollout_every: int = 50        # steps between rollout generations
    updates_per_step: int = 20
    real_ratio: float = 0.05       # fraction of real transitions in mixed batch
    model_retain_epochs: int = 1   # retain roughly this many epochs of imagined rollouts
    env_buffer_capacity: int = 1_000_000
    model_buffer_capacity: int = 400_000
    num_warmup_steps: int = 1_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.real_ratio <= 1.0:
            raise ValueError(
                f"real_ratio must be in [0, 1], got {self.real_ratio}."
            )
        if self.ensemble_size < 2:
            raise ValueError(
                f"ensemble_size must be at least 2, got {self.ensemble_size}."
            )
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if self.model_hidden_dim <= 0:
            raise ValueError("model_hidden_dim must be positive.")
        if self.model_n_layers < 1:
            raise ValueError("model_n_layers must be at least 1.")
        if self.model_lr <= 0.0:
            raise ValueError("model_lr must be positive.")
        if self.model_fit_steps <= 0:
            raise ValueError("model_fit_steps must be positive.")
        if self.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.actor_lr <= 0.0:
            raise ValueError("actor_lr must be positive.")
        if self.critic_lr <= 0.0:
            raise ValueError("critic_lr must be positive.")
        if not 0.0 < self.tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {self.tau}.")
        if self.sac_hidden_dim <= 0:
            raise ValueError("sac_hidden_dim must be positive.")
        if self.sac_batch_size <= 0:
            raise ValueError("sac_batch_size must be positive.")
        if self.rollout_length <= 0:
            raise ValueError("rollout_length must be positive.")
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive.")
        if self.rollout_every <= 0:
            raise ValueError("rollout_every must be positive.")
        if self.updates_per_step <= 0:
            raise ValueError("updates_per_step must be positive.")
        if self.model_retain_epochs <= 0:
            raise ValueError("model_retain_epochs must be positive.")
        if self.env_buffer_capacity <= 0:
            raise ValueError("env_buffer_capacity must be positive.")
        if self.model_buffer_capacity <= 0:
            raise ValueError("model_buffer_capacity must be positive.")
        if self.num_warmup_steps <= 0:
            raise ValueError("num_warmup_steps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MbpoConfig":
        """Construct a config from a plain dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
