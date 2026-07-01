"""Configuration for DDPG and TD3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict

_VALID_NOISE_TYPES = {"gaussian", "ou"}


@register_config("ddpg")
@dataclass
class DDPGConfig(BaseConfig):
    """Configuration for DDPG training (Deep Deterministic Policy Gradient).

    Optimized by default for continuous MuJoCo-style environments
    (HalfCheetah-v5). Inherits from ``BaseConfig`` and adds the hyperparameters
    specific to off-policy deterministic actor-critic.

    The DDPG algorithm (Lillicrap et al., 2015) combines:
    - A deterministic policy μ_θ(s) updated via the gradient of Q.
    - A critic Q_φ(s, a) trained by Bellman regression.
    - Target networks (actor_target, critic_target) updated softly.
    - A replay buffer for data decorrelation.
    """

    # Approach identifier
    approach: str = "ddpg"

    # Override BaseConfig defaults for MuJoCo
    env_id: str = "HalfCheetah-v5"
    gamma: float = 0.99
    total_timesteps: int = 1_000_000
    max_steps_per_episode: int = 1000

    # Network architecture
    hidden_dim: int = 256

    # Separate learning rates for actor and critic
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    # Soft update of the target networks: θ_target ← τ·θ + (1-τ)·θ_target
    tau: float = 0.005

    # Off-policy replay buffer
    buffer_capacity: int = 1_000_000
    batch_size: int = 256

    # Warm-up: random actions during the first start_steps steps
    start_steps: int = 10_000

    # Start of updates and frequency
    update_after: int = 1_000
    update_every: int = 1

    # Exploration noise
    noise_type: str = "gaussian"  # "gaussian" or "ou"
    noise_std: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.start_steps < 0:
            raise ValueError("start_steps must be non-negative.")
        if self.update_after < 0:
            raise ValueError("update_after must be non-negative.")
        if self.update_every <= 0:
            raise ValueError("update_every must be positive.")
        if self.noise_type not in _VALID_NOISE_TYPES:
            raise ValueError(
                f"noise_type must be one of {_VALID_NOISE_TYPES}, got '{self.noise_type}'."
            )
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.actor_lr <= 0:
            raise ValueError("actor_lr must be positive.")
        if self.critic_lr <= 0:
            raise ValueError("critic_lr must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DDPGConfig:
        """Build from a dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)


@register_config("td3")
@dataclass
class TD3Config(DDPGConfig):
    """Configuration for TD3 training (Twin Delayed DDPG).

    Extends ``DDPGConfig`` with the three key improvements of TD3
    (Fujimoto et al., 2018):
    1. Twin critics — min(Q1, Q2) for the target to reduce overestimation.
    2. Delayed policy updates — the actor is updated less often than the critic.
    3. Target policy smoothing — clipped noise added to the target action to regularize.
    """

    # Approach identifier
    approach: str = "td3"

    # Delayed actor update: 1 actor update per policy_delay critic updates
    policy_delay: int = 2

    # Target policy smoothing: clipped noise added to the TD3 target action
    target_noise: float = 0.2
    target_noise_clip: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.policy_delay <= 0:
            raise ValueError("policy_delay must be positive.")
        if self.target_noise < 0:
            raise ValueError("target_noise must be non-negative.")
        if self.target_noise_clip < 0:
            raise ValueError("target_noise_clip must be non-negative.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TD3Config:
        """Build from a dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
