"""Configuration for A2C, A2C-GAE, and A3C."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("a2c")
@dataclass
class A2CConfig(BaseConfig):
    """Configuration for A2C training (Advantage Actor-Critic).

    Optimized by default for continuous MuJoCo-style environments
    (HalfCheetah-v5, Hopper-v5, etc.).
    """

    # Override the default values from BaseConfig
    approach: str = "a2c"
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    gamma: float = 0.99
    checkpoint_every: int = 50_000

    # Hyperparameters specific to A2C
    hidden_dim: int = 256
    lr: float = 3e-4
    n_steps: int = 2048
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Observation normalization
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> A2CConfig:
        """Build from a dict and reject unknown keys."""
        return strict_dataclass_from_dict(cls, payload)


@register_config("a2c_gae")
@dataclass
class A2CGAEConfig(BaseConfig):
    """Configuration for A2C-GAE training (A2C with GAE).

    Identical to ``A2CConfig`` with the additional ``gae_lambda`` parameter
    that controls the exponential smoothing of GAE.
    """

    # Override the default values from BaseConfig
    approach: str = "a2c_gae"
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    gamma: float = 0.99
    checkpoint_every: int = 50_000

    # Hyperparameters shared with A2C
    hidden_dim: int = 256
    lr: float = 3e-4
    n_steps: int = 2048
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Observation normalization
    normalize_observations: bool = False
    obs_norm_epsilon: float = 1e-8
    obs_norm_clip: float = 10.0

    # Hyperparameter specific to GAE
    gae_lambda: float = 0.95

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
        """Serialize the config to a Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> A2CGAEConfig:
        """Build from a dict and reject unknown keys."""
        return strict_dataclass_from_dict(cls, payload)


@register_config("a3c")
@dataclass
class A3CConfig(A2CGAEConfig):
    """Configuration for A3C training (Asynchronous Advantage Actor-Critic).

    Extends ``A2CGAEConfig`` with the hyperparameters specific to asynchronous
    parallelism: number of workers and rollout length per worker (t_max).

    Optimized by default for continuous non-MuJoCo environments
    (Pendulum-v1, LunarLanderContinuous-v2, etc.).
    """

    # Approach identifier
    approach: str = "a3c"

    # Hyperparameters specific to A3C
    num_workers: int = 4
    t_max: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive.")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> A3CConfig:
        """Build from a dict and reject unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
