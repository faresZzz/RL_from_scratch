"""Configuration for SAC (Soft Actor-Critic)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("sac")
@dataclass
class SACConfig(BaseConfig):
    """Configuration for SAC (Soft Actor-Critic) training.

    Optimized by default for continuous MuJoCo-style environments
    (HalfCheetah-v5). Inherits from ``BaseConfig`` and adds the hyperparameters
    specific to off-policy stochastic actor-critic with maximum entropy.

    The SAC algorithm (Haarnoja et al., 2018) combines:
    - A squashed Gaussian stochastic policy π_θ(a|s).
    - Twin critics Q_φ(s, a) trained by Bellman regression.
    - An entropy term α·H(π) in the objective for exploration.
    - Automatic tuning of α via the target entropy constraint.
    - A replay buffer for decorrelating the data.
    """

    # Approach identification
    approach: str = "sac"

    # Override BaseConfig defaults for MuJoCo
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    max_steps_per_episode: int = 1000

    # Network architecture
    hidden_dim: int = 256

    # Separate learning rates for actor and critic
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Discount factor
    gamma: float = 0.99

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

    # Entropy coefficient
    alpha: float = 0.2
    auto_tune_alpha: bool = True
    alpha_lr: float = 3e-4
    target_entropy: float | None = None  # None → -action_dim

    # Bounds of the Gaussian actor's log standard deviation
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be in (0, 1).")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive.")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be strictly less than log_std_max.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config into a Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SACConfig:
        """Build from a dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
