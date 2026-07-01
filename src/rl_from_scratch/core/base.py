"""Base classes for all RL agents and configs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class BaseConfig:
    """Shared configuration fields for every RL algorithm."""

    env_id: str = "CartPole-v1"
    seed: int = 0
    total_timesteps: int = 100_000
    episodes: int | None = None
    max_steps_per_episode: int = 1000
    gamma: float = 0.99
    output_dir: str = "runs"
    run_name: str | None = None
    approach: str = "q_learning"
    checkpoint_every: int = 50
    checkpoint_keep_last: int = 3
    eval_every: int = 10
    eval_every_steps: int | None = None
    eval_episodes: int = 5
    eval_seed: int = 10_000
    num_seeds: int = 1
    render: bool = False
    record_every: int | None = None
    device: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)
    solved_reward: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")
        if self.episodes is not None and self.episodes <= 0:
            raise ValueError("episodes must be positive when set.")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive.")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")
        if self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive.")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive.")
        if self.eval_every_steps is not None and self.eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive when set.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive.")
        if self.eval_seed < 0:
            raise ValueError("eval_seed must be non-negative.")
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be at least 1.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.record_every is not None and self.record_every <= 0:
            raise ValueError("record_every must be positive when set.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BaseConfig:
        """Construct a config from a plain dict, rejecting unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - known)
        if unknown:
            keys = ", ".join(unknown)
            raise ValueError(f"Unknown config keys for {cls.__name__}: {keys}")
        return cls(**payload)


class BaseAgent(ABC):
    """Abstract base for every RL agent implementation."""

    @abstractmethod
    def select_action(self, observation: Any, *, deterministic: bool = False) -> Any:
        """Choose an action given the current observation."""

    @abstractmethod
    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Run one learning update; return a dict of loss/metric values."""

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Store a transition for experience-replay agents (default no-op)."""

    def episode_ended(self) -> None:
        """Hook called at the end of every episode (default no-op)."""

    def _to_tensor(self, obs: Any) -> torch.Tensor:
        """Convert an observation to a batched float32 tensor on ``self.device``.

        Adds a leading batch dimension to 1-D inputs. Available to every
        Torch-based agent; tabular agents simply never call it.
        """
        import torch

        tensor = torch.as_tensor(obs, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Persist agent state to *path* and return the written path."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path, **kwargs: Any) -> BaseAgent:
        """Restore an agent from a previously saved checkpoint."""
