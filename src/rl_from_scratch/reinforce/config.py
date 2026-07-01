"""Configuration for REINFORCE and REINFORCE with baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("reinforce")
@dataclass
class ReinforceConfig(BaseConfig):
    """Configuration for REINFORCE training (Monte Carlo Policy Gradient)."""

    # Override BaseConfig defaults
    approach: str = "reinforce"
    gamma: float = 0.99
    episodes: int = 1000  # type: ignore[assignment]

    # REINFORCE-specific fields
    hidden_dim: int = 64
    lr: float = 1e-3

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.episodes is not None and self.episodes <= 0:
            raise ValueError("episodes must be positive.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")
        if self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config into a Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReinforceConfig:
        """Build from a dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)


@register_config("reinforce_baseline")
@dataclass
class ReinforceBaselineConfig(BaseConfig):
    """Configuration for REINFORCE with a value baseline."""

    # Override BaseConfig defaults
    approach: str = "reinforce_baseline"
    gamma: float = 0.99
    episodes: int = 1000  # type: ignore[assignment]

    # Fields specific to REINFORCE with baseline
    hidden_dim: int = 64
    lr_policy: float = 1e-3
    lr_value: float = 1e-3

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.episodes is not None and self.episodes <= 0:
            raise ValueError("episodes must be positive.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.lr_policy <= 0:
            raise ValueError("lr_policy must be positive.")
        if self.lr_value <= 0:
            raise ValueError("lr_value must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")
        if self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config into a Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReinforceBaselineConfig:
        """Build from a dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
