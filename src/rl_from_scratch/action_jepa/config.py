"""Configuration for Action-JEPA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("action_jepa")
@dataclass
class ActionJepaConfig(BaseConfig):
    """Configuration for an action-conditioned JEPA world model."""

    approach: str = "action_jepa"
    env_id: str = "Pendulum-v1"
    episodes: int = 10  # type: ignore[assignment]
    max_steps_per_episode: int = 200

    latent_dim: int = 32
    hidden_dim: int = 256
    encoder_layers: int = 2

    training_regime: str = "joint"
    freeze_encoder_after_pretrain: bool = False
    random_frozen_encoder: bool = False
    mask_fraction: float = 1.0 / 3.0
    mask_noise_std: float = 0.01
    representation_val_every: int = 10

    ema_tau: float = 0.99
    predictor_delta: bool = True
    rollout_len: int = 5
    reward_loss_weight: float = 1.0
    continuation_loss_weight: float = 1.0
    variance_loss_weight: float = 1.0
    covariance_loss_weight: float = 0.04
    target_std: float = 1.0

    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 10.0
    batch_size: int = 128
    buffer_capacity: int = 200_000
    learning_starts: int = 2_000

    num_warmup_steps: int = 2_000
    pretrain_steps: int = 250
    collect_every: int = 100
    updates_per_collect: int = 25
    control_updates_per_step: int = 1

    plan_mode: str = "goal"
    goal_obs: list[float] | None = field(default_factory=lambda: [1.0, 0.0, 0.0])
    plan_horizon: int = 12
    cem_population: int = 256
    cem_num_elites: int = 32
    cem_iterations: int = 4
    cem_alpha: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.encoder_layers < 1:
            raise ValueError("encoder_layers must be at least 1.")
        allowed_regimes = {"joint", "stage-wise", "stage-wise-fair", "random-frozen"}
        if self.training_regime not in allowed_regimes:
            raise ValueError(
                f"training_regime must be one of {sorted(allowed_regimes)}, "
                f"got {self.training_regime!r}."
            )
        if self.training_regime == "joint" and self.freeze_encoder_after_pretrain:
            raise ValueError("joint training must not freeze the encoder.")
        if self.training_regime in {"stage-wise", "stage-wise-fair"} and not self.freeze_encoder_after_pretrain:
            raise ValueError("stage-wise regimes require freeze_encoder_after_pretrain=True.")
        if self.training_regime == "random-frozen" and not self.random_frozen_encoder:
            raise ValueError("random-frozen requires random_frozen_encoder=True.")
        if self.training_regime != "random-frozen" and self.random_frozen_encoder:
            raise ValueError("random_frozen_encoder is only valid for random-frozen.")
        if not 0.0 < self.mask_fraction < 1.0:
            raise ValueError("mask_fraction must be in (0, 1).")
        if self.mask_noise_std < 0.0:
            raise ValueError("mask_noise_std must be non-negative.")
        if self.representation_val_every <= 0:
            raise ValueError("representation_val_every must be positive.")
        if not 0.0 < self.ema_tau < 1.0:
            raise ValueError("ema_tau must be in (0, 1).")
        if self.rollout_len < 1:
            raise ValueError("rollout_len must be at least 1.")
        for name in (
            "reward_loss_weight",
            "continuation_loss_weight",
            "variance_loss_weight",
            "covariance_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.target_std <= 0.0:
            raise ValueError("target_std must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")
        if self.learning_starts <= 0:
            raise ValueError("learning_starts must be positive.")
        if self.num_warmup_steps <= 0:
            raise ValueError("num_warmup_steps must be positive.")
        if self.pretrain_steps < 0:
            raise ValueError("pretrain_steps must be non-negative.")
        if self.collect_every <= 0:
            raise ValueError("collect_every must be positive.")
        if self.updates_per_collect <= 0:
            raise ValueError("updates_per_collect must be positive.")
        if self.control_updates_per_step <= 0:
            raise ValueError("control_updates_per_step must be positive.")
        if self.plan_mode not in {"goal", "reward"}:
            raise ValueError("plan_mode must be 'goal' or 'reward'.")
        if self.plan_mode == "goal" and self.goal_obs is None:
            raise ValueError("goal_obs is required when plan_mode='goal'.")
        if self.plan_horizon <= 0:
            raise ValueError("plan_horizon must be positive.")
        if self.cem_population <= 0:
            raise ValueError("cem_population must be positive.")
        if self.cem_num_elites <= 0 or self.cem_num_elites > self.cem_population:
            raise ValueError("cem_num_elites must be in [1, cem_population].")
        if self.cem_iterations <= 0:
            raise ValueError("cem_iterations must be positive.")
        if not 0.0 < self.cem_alpha <= 1.0:
            raise ValueError("cem_alpha must be in (0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionJepaConfig":
        return strict_dataclass_from_dict(cls, payload)
