"""Configuration for PETS (Probabilistic Ensembles with Trajectory Sampling).

PETS is an episode-based model-based reinforcement learning algorithm (Chua et al. 2018).
Each iteration consists of:

1. Fitting a probabilistic ensemble of neural network dynamics models on all
   observed transitions (s, a) -> Delta-s using maximum-likelihood estimation.

2. Planning with Cross-Entropy Method (CEM) over the ensemble using particle-based
   trajectory sampling (TS1 or TS-infinity) to select actions online.

3. Collecting one real episode in the environment using CEM planning at every step.

The ``episodes`` field counts PETS iterations (= real environment episodes), not timesteps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("pets")
@dataclass
class PetsConfig(BaseConfig):
    """Configuration for PETS on HalfCheetah-v5 (or any continuous env)."""

    approach: str = "pets"
    env_id: str = "HalfCheetah-v5"
    episodes: int = 30  # type: ignore[assignment]  # PETS iterations (= real episodes)
    max_steps_per_episode: int = 1000

    # ── Probabilistic ensemble architecture ──────────────────────────────
    ensemble_size: int = 5          # Number of independent dynamics networks
    hidden_dim: int = 200           # Width of each hidden layer
    n_layers: int = 3               # Number of hidden layers

    # ── Ensemble training ─────────────────────────────────────────────────
    dynamics_lr: float = 1e-3       # Adam learning rate for dynamics
    dynamics_fit_steps: int = 100   # Adam steps per episode
    dynamics_batch_size: int = 256  # Minibatch size for dynamics training
    weight_decay: float = 1e-4      # L2 regularisation weight

    # ── CEM planner ───────────────────────────────────────────────────────
    plan_horizon: int = 25          # Planning horizon (steps)
    cem_population: int = 400       # CEM candidate population size
    cem_elite_frac: float = 0.1     # Fraction of elites (top-k candidates)
    cem_iterations: int = 5         # CEM refinement iterations
    cem_alpha: float = 0.1          # CEM momentum (belief update weight)
    risk_beta: float = 0.0          # Penalise particle-return dispersion

    # ── Trajectory sampling ───────────────────────────────────────────────
    n_particles: int = 20           # Particles per candidate for plan rollout
    ts_mode: str = "tsinf"          # "ts1" or "tsinf" trajectory sampling

    # ── Warm-up ───────────────────────────────────────────────────────────
    num_warmup_steps: int = 1000    # Random steps before first planning

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ts_mode not in {"ts1", "tsinf"}:
            raise ValueError(
                f"ts_mode must be 'ts1' or 'tsinf', got '{self.ts_mode}'."
            )
        if not 0.0 < self.cem_elite_frac <= 1.0:
            raise ValueError(
                f"cem_elite_frac must be in (0, 1], got {self.cem_elite_frac}."
            )
        if self.ensemble_size < 2:
            raise ValueError(
                f"ensemble_size must be at least 2, got {self.ensemble_size}."
            )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if self.dynamics_lr <= 0.0:
            raise ValueError("dynamics_lr must be positive.")
        if self.dynamics_fit_steps <= 0:
            raise ValueError("dynamics_fit_steps must be positive.")
        if self.dynamics_batch_size <= 0:
            raise ValueError("dynamics_batch_size must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.plan_horizon <= 0:
            raise ValueError("plan_horizon must be positive.")
        if self.cem_population <= 0:
            raise ValueError("cem_population must be positive.")
        if self.cem_iterations <= 0:
            raise ValueError("cem_iterations must be positive.")
        if self.risk_beta < 0.0:
            raise ValueError("risk_beta must be non-negative.")
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive.")
        if self.num_warmup_steps <= 0:
            raise ValueError("num_warmup_steps must be positive.")
        # CEM needs >= 2 elites to fit a std (n_elites = ceil(frac * population));
        # frac * population <= 1 collapses the refit to a NaN std.
        if self.cem_elite_frac * self.cem_population <= 1.0:
            raise ValueError(
                "cem_elite_frac * cem_population must exceed 1 (need >= 2 CEM elites)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a plain dict suitable for JSON/YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PetsConfig":
        """Construct a config from a plain dict, rejecting unknown keys."""
        return strict_dataclass_from_dict(cls, payload)
