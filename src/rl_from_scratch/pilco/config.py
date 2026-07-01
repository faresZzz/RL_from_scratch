"""Configuration for PILCO (Probabilistic Inference for Learning COntrol).

PILCO is an episode-based model-based policy search algorithm.  Each
*episode* (i.e. PILCO iteration) consists of:

1. Fitting a GP dynamics model on all observed transitions.
2. Optimising the RBF policy by back-propagating through the analytic
   trajectory prediction.
3. Rolling out one real episode with the improved policy.

The ``episodes`` field therefore counts PILCO iterations, not timesteps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import register_config, strict_dataclass_from_dict


@register_config("pilco")
@dataclass
class PilcoConfig(BaseConfig):
    """Configuration for PILCO on Pendulum-v1 (or any continuous env)."""

    approach: str = "pilco"
    env_id: str = "Pendulum-v1"
    episodes: int = 15  # type: ignore[assignment]  # PILCO iterations
    max_steps_per_episode: int = 200
    collection_horizon: int | None = None

    # Policy representational capacity
    n_basis: int = 50
    policy_type: str = "rbf"
    policy_hidden_dim: int = 64
    policy_hidden_layers: int = 2

    # Planning horizon (number of analytic belief-propagation steps)
    horizon: int = 30

    # GP hyperparameter fitting (L-BFGS max iterations)
    gp_fit_steps: int = 50

    # Policy optimisation steps (L-BFGS max iterations)
    policy_opt_steps: int = 50
    policy_lr: float = 0.3

    # Approximate a non-Gaussian reset distribution by several Gaussian beliefs.
    n_initial_beliefs: int = 1

    # Number of initial random rollouts before the first GP fit
    num_init_rollouts: int = 1

    # Subsample cap for GP tractability (random subset when buffer > cap)
    max_gp_points: int = 300

    # Initial belief covariance diagonal σ₀² * I
    init_state_cov: float = 0.01

    # Diagonal weights W for the saturating cost (one per state dim)
    # Pendulum: state = [cosθ, sinθ, θ̇]  → W = diag(1, 1, 0.1)
    cost_weight: tuple = (1.0, 1.0, 0.1)  # type: ignore[assignment]

    # Gaussian noise added to policy actions during exploration
    exploration_noise: float = 0.0
    cost_mode: str = "saturating"
    terminal_penalty: float = 10.0
    action_cost_weight: float = 1e-4
    validation_fraction: float = 0.15
    validation_min_points: int = 32
    final_eval_episodes: int = 20
    final_eval_seed: int = 20_000

    # ── Angle encoding (InvertedPendulum recipe) ──────────────────────────
    # When True, raw 4-D obs [cart_pos, θ, cart_vel, θdot] is mapped to
    # 5-D encoded state [cart_pos, sinθ, cosθ, cart_vel, θdot] at the env
    # boundary.  The GP, policy, and cost all operate on the 5-D space.
    encode_angle: bool = False

    # ── Fixed-horizon data collection ─────────────────────────────────────
    # When > 0, data-collection rollouts run for exactly this many steps
    # regardless of early termination (using NoEarlyTermination wrapper).
    # Set to match the imagined horizon so the GP sees the same time scale.
    fixed_horizon_steps: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_basis <= 0:
            raise ValueError("n_basis must be positive.")
        if self.policy_type not in {"rbf", "linear_sine"}:
            raise ValueError("policy_type must be 'rbf' or 'linear_sine'.")
        if self.policy_hidden_dim <= 0:
            raise ValueError("policy_hidden_dim must be positive.")
        if self.policy_hidden_layers <= 0:
            raise ValueError("policy_hidden_layers must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.gp_fit_steps <= 0:
            raise ValueError("gp_fit_steps must be positive.")
        if self.policy_opt_steps <= 0:
            raise ValueError("policy_opt_steps must be positive.")
        if self.policy_lr <= 0.0:
            raise ValueError("policy_lr must be positive.")
        if self.n_initial_beliefs <= 0:
            raise ValueError("n_initial_beliefs must be positive.")
        if self.num_init_rollouts <= 0:
            raise ValueError("num_init_rollouts must be positive.")
        if self.max_gp_points <= 0:
            raise ValueError("max_gp_points must be positive.")
        if self.init_state_cov <= 0.0:
            raise ValueError("init_state_cov must be positive.")
        if len(self.cost_weight) == 0:
            raise ValueError("cost_weight must be non-empty.")
        if any(w < 0.0 for w in self.cost_weight):
            raise ValueError("all cost_weight entries must be non-negative.")
        if all(w == 0.0 for w in self.cost_weight):
            raise ValueError("at least one cost_weight entry must be positive.")
        if self.exploration_noise < 0.0:
            raise ValueError("exploration_noise must be non-negative.")
        if self.cost_mode not in {"saturating", "inverted_pendulum"}:
            raise ValueError("cost_mode must be 'saturating' or 'inverted_pendulum'.")
        if self.terminal_penalty < 0.0:
            raise ValueError("terminal_penalty must be non-negative.")
        if self.action_cost_weight < 0.0:
            raise ValueError("action_cost_weight must be non-negative.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.validation_min_points < 0:
            raise ValueError("validation_min_points must be non-negative.")
        if self.final_eval_episodes <= 0:
            raise ValueError("final_eval_episodes must be positive.")
        if self.final_eval_seed < 0:
            raise ValueError("final_eval_seed must be non-negative.")
        if self.fixed_horizon_steps < 0:
            raise ValueError("fixed_horizon_steps must be non-negative.")
        if self.collection_horizon is not None and self.collection_horizon <= 0:
            raise ValueError("collection_horizon must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost_weight"] = list(self.cost_weight)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilcoConfig:
        return strict_dataclass_from_dict(
            cls,
            payload,
            converters={"cost_weight": lambda values: tuple(float(v) for v in values)},
        )


@register_config("deep_pilco")
@dataclass
class DeepPilcoConfig(BaseConfig):
    """Configuration for Deep PILCO on Pendulum-v1 (or any continuous env).

    Deep PILCO replaces the GP dynamics model with a Bayesian neural network
    approximated via MC-dropout, and replaces analytic moment matching with
    particle-based propagation + empirical Gaussian resampling.

    Each episode (i.e. Deep PILCO iteration) follows the same three-phase
    structure as PILCO:

    1. **BNN fit**: train the BNN dynamics on all buffered (s, a) → Δs transitions
       using Adam + MSE loss, with stochastic dropout active.
    2. **Policy optimisation**: optimise the RBF policy by back-propagating
       through ``predict_trajectory_particles`` with Adam.
    3. **Real rollout**: collect one episode with the improved policy.
    """

    approach: str = "deep_pilco"
    env_id: str = "Pendulum-v1"
    episodes: int = 15  # type: ignore[assignment]  # Deep PILCO iterations

    max_steps_per_episode: int = 200
    collection_horizon: int | None = None

    # ── BNN architecture ──────────────────────────────────────────────────
    hidden_dim: int = 64            # Width of each hidden layer
    n_layers: int = 2               # Number of hidden layers
    dropout_p: float = 0.05         # MC-dropout probability p

    # ── Particle propagation ──────────────────────────────────────────────
    n_particles: int = 30           # Number of particles K for trajectory prediction
    horizon: int = 25               # Planning horizon (number of propagation steps)

    # ── BNN training ──────────────────────────────────────────────────────
    model_train_steps: int = 100    # Adam steps per episode for BNN
    model_batch_size: int = 64      # Minibatch size for BNN training
    model_lr: float = 1e-3         # Adam learning rate for BNN

    # ── Policy optimisation ───────────────────────────────────────────────
    policy_opt_steps: int = 50      # Adam steps per episode for policy
    policy_lr: float = 0.01         # Adam learning rate for policy

    # ── Policy representation ─────────────────────────────────────────────
    n_basis: int = 50               # Number of RBF centres in the policy
    policy_type: str = "rbf"
    policy_hidden_dim: int = 64
    policy_hidden_layers: int = 2

    # ── Buffer / data ──────────────────────────────────────────────────────
    num_init_rollouts: int = 1      # Random rollouts before the first BNN fit
    max_gp_points: int = 300        # Buffer cap (same field name as PilcoConfig)

    # ── Initial belief ────────────────────────────────────────────────────
    init_state_cov: float = 0.01    # Σ₀ = init_state_cov * I

    # ── Cost ──────────────────────────────────────────────────────────────
    # Pendulum: state = [cosθ, sinθ, θ̇]  → W = diag(1, 1, 0.1)
    cost_weight: tuple = (1.0, 1.0, 0.1)  # type: ignore[assignment]

    # ── Exploration ───────────────────────────────────────────────────────
    exploration_noise: float = 0.0  # Std of additive noise on actions
    cost_mode: str = "saturating"
    validation_fraction: float = 0.15
    validation_min_points: int = 32
    final_eval_episodes: int = 20
    final_eval_seed: int = 20_000

    # ── Angle encoding (InvertedPendulum recipe) ────────────────────
    # When True, raw 4-D obs [cart_pos, θ, cart_vel, θdot] is mapped to
    # 5-D encoded state [cart_pos, sinθ, cosθ, cart_vel, θdot].
    # The BNN, policy, and cost all operate on the 5-D space.
    encode_angle: bool = False

    # ── Fixed-horizon data collection ─────────────────────────────
    # When > 0, data-collection rollouts use NoEarlyTermination wrapper
    # so the BNN sees pole-falling transitions (real dynamics variety).
    fixed_horizon_steps: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1).")
        if self.n_particles <= 1:
            raise ValueError("n_particles must be > 1.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.model_train_steps <= 0:
            raise ValueError("model_train_steps must be positive.")
        if self.model_batch_size <= 0:
            raise ValueError("model_batch_size must be positive.")
        if self.model_lr <= 0.0:
            raise ValueError("model_lr must be positive.")
        if self.policy_opt_steps <= 0:
            raise ValueError("policy_opt_steps must be positive.")
        if self.policy_lr <= 0.0:
            raise ValueError("policy_lr must be positive.")
        if self.n_basis <= 0:
            raise ValueError("n_basis must be positive.")
        if self.policy_type not in {"rbf", "mlp"}:
            raise ValueError("policy_type must be 'rbf' or 'mlp'.")
        if self.policy_hidden_dim <= 0:
            raise ValueError("policy_hidden_dim must be positive.")
        if self.policy_hidden_layers <= 0:
            raise ValueError("policy_hidden_layers must be positive.")
        if self.num_init_rollouts <= 0:
            raise ValueError("num_init_rollouts must be positive.")
        if self.max_gp_points <= 0:
            raise ValueError("max_gp_points must be positive.")
        if self.init_state_cov <= 0.0:
            raise ValueError("init_state_cov must be positive.")
        if len(self.cost_weight) == 0:
            raise ValueError("cost_weight must be non-empty.")
        if any(w < 0.0 for w in self.cost_weight):
            raise ValueError("all cost_weight entries must be non-negative.")
        if all(w == 0.0 for w in self.cost_weight):
            raise ValueError("at least one cost_weight entry must be positive.")
        if self.exploration_noise < 0.0:
            raise ValueError("exploration_noise must be non-negative.")
        if self.cost_mode not in {"saturating", "inverted_pendulum"}:
            raise ValueError("cost_mode must be 'saturating' or 'inverted_pendulum'.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.validation_min_points < 0:
            raise ValueError("validation_min_points must be non-negative.")
        if self.final_eval_episodes <= 0:
            raise ValueError("final_eval_episodes must be positive.")
        if self.final_eval_seed < 0:
            raise ValueError("final_eval_seed must be non-negative.")
        if self.fixed_horizon_steps < 0:
            raise ValueError("fixed_horizon_steps must be non-negative.")
        if self.collection_horizon is not None and self.collection_horizon <= 0:
            raise ValueError("collection_horizon must be positive when provided.")
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost_weight"] = list(self.cost_weight)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeepPilcoConfig":
        return strict_dataclass_from_dict(
            cls,
            payload,
            converters={"cost_weight": lambda values: tuple(float(v) for v in values)},
        )
