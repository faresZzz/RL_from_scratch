"""Configuration for DreamerV1 (Dreamer: Dream to Control, Hafner et al. 2020).

DreamerV1 learns a latent world model (RSSM) and optimises behaviours entirely
in imagination:
1. Encode observations → compact embedding.
2. RSSM transitions (recurrent + stochastic) → posterior/prior latent states.
3. Decode latents back to observations + predict reward.
4. Imagine H-step rollouts in latent space; maximise λ-returns w.r.t. actor.
5. Critic regresses to the imagined returns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("dreamer")
@dataclass
class DreamerConfig(BaseConfig):
    """Configuration for DreamerV1 on HalfCheetah-v5 (or any continuous env)."""

    approach: str = "dreamer"
    env_id: str = "HalfCheetah-v5"
    epochs: int = 30
    steps_per_epoch: int = 1_000
    max_steps_per_episode: int = 1_000

    # ── RSSM / world-model architecture ──────────────────────────────────
    deter_dim: int = 200
    stoch_dim: int = 30
    rssm_hidden_dim: int = 200
    encoder_hidden_dim: int = 200
    decoder_hidden_dim: int = 200
    reward_hidden_dim: int = 200
    embed_dim: int = 200

    # ── World-model training ──────────────────────────────────────────────
    model_lr: float = 6e-4
    free_nats: float = 3.0
    kl_scale: float = 1.0
    grad_clip: float = 100.0
    min_std: float = 0.1

    # ── Behaviour (actor-critic in imagination) ───────────────────────────
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    lambda_: float = 0.95
    imagination_horizon: int = 15
    actor_entropy: float = 1e-4

    # ── Training schedule ─────────────────────────────────────────────────
    batch_size: int = 32
    batch_length: int = 32
    train_every: int = 1
    num_warmup_steps: int = 1_000
    buffer_capacity: int = 200_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.deter_dim <= 0:
            raise ValueError("deter_dim must be positive.")
        if self.stoch_dim <= 0:
            raise ValueError("stoch_dim must be positive.")
        if self.rssm_hidden_dim <= 0:
            raise ValueError("rssm_hidden_dim must be positive.")
        if self.encoder_hidden_dim <= 0:
            raise ValueError("encoder_hidden_dim must be positive.")
        if self.decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive.")
        if self.reward_hidden_dim <= 0:
            raise ValueError("reward_hidden_dim must be positive.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.model_lr <= 0.0:
            raise ValueError("model_lr must be positive.")
        if self.free_nats < 0.0:
            raise ValueError("free_nats must be non-negative.")
        if self.kl_scale < 0.0:
            raise ValueError("kl_scale must be non-negative.")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive.")
        if self.min_std <= 0.0:
            raise ValueError("min_std must be positive.")
        if self.actor_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim must be positive.")
        if self.critic_hidden_dim <= 0:
            raise ValueError("critic_hidden_dim must be positive.")
        if self.actor_lr <= 0.0:
            raise ValueError("actor_lr must be positive.")
        if self.critic_lr <= 0.0:
            raise ValueError("critic_lr must be positive.")
        if not (0 < self.lambda_ <= 1):
            raise ValueError(
                f"lambda_ must be in (0, 1], got {self.lambda_}."
            )
        if self.imagination_horizon <= 0:
            raise ValueError(
                f"imagination_horizon must be positive, got {self.imagination_horizon}."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.batch_length < 2:
            raise ValueError(
                f"batch_length must be >= 2, got {self.batch_length}."
            )
        if self.train_every < 1:
            raise ValueError(
                f"train_every must be >= 1, got {self.train_every}."
            )
        if self.num_warmup_steps <= 0:
            raise ValueError("num_warmup_steps must be positive.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DreamerConfig":
        """Construct a config from a plain dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
