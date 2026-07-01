"""PETS agent — Probabilistic Ensembles with Trajectory Sampling.

The agent wraps:
- A ``ProbabilisticEnsemble`` for dynamics modelling.
- A ``CEMPlanner`` for action selection via trajectory sampling + CEM.
- A ``TransitionBuffer`` for replay data.

Action selection has two phases:
1. **Warm-up** (``len(buffer) < num_warmup_steps``): uniform random actions to
   seed the ensemble with enough data for meaningful dynamics fitting.
2. **Planning** (after warm-up): CEM over the ensemble with warm-start from
   the previous step's optimal mean (action-sequence shift).

The dynamics model is re-fitted at the start of each episode via ``learn_step``,
which trains on the full collected buffer.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device
from rl_from_scratch.pets.buffer import TransitionBuffer
from rl_from_scratch.pets.dynamics import ProbabilisticEnsemble
from rl_from_scratch.pets.planner import CEMPlanner
from rl_from_scratch.pets.reward import get_reward_fn


class PetsAgent(BaseAgent):
    """PETS: model-based RL via probabilistic ensemble + CEM trajectory sampling.

    Parameters
    ----------
    obs_dim:
        Observation space dimensionality.
    action_dim:
        Action space dimensionality.
    action_low:
        Per-dimension lower bounds on actions (array or list).
    action_high:
        Per-dimension upper bounds on actions (array or list).
    env_id:
        Gymnasium environment ID (used to look up the reward function).
    reward_dt:
        Environment timestep (``env.unwrapped.dt``), passed to the reward fn.
    ensemble_size:
        Number of independent probabilistic MLP dynamics models.
    hidden_dim:
        Width of each hidden layer in the dynamics networks.
    n_layers:
        Number of hidden layers in the dynamics networks.
    dynamics_lr:
        Adam learning rate for dynamics training.
    dynamics_fit_steps:
        Adam steps per episode for dynamics fitting.
    dynamics_batch_size:
        Mini-batch size during dynamics training.
    weight_decay:
        L2 regularisation coefficient for Adam.
    plan_horizon:
        CEM planning horizon (steps).
    cem_population:
        CEM candidate population size.
    cem_elite_frac:
        Fraction of elites for CEM belief update.
    cem_iterations:
        CEM refinement iterations per planning call.
    cem_alpha:
        CEM momentum coefficient (belief update step size).
    risk_beta:
        Penalty on the standard deviation of particle returns in CEM.  A
        positive value makes planning more conservative under ensemble
        disagreement.
    n_particles:
        Particles per candidate in the CEM rollout.
    ts_mode:
        ``"ts1"`` or ``"tsinf"`` trajectory sampling mode.
    num_warmup_steps:
        Number of random steps before switching to CEM planning.
    gamma:
        Discount factor (accepted for build_agent compatibility; unused).
    seed:
        Random seed for the ensemble and planner.
    device:
        Compute device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        action_low: Any,
        action_high: Any,
        env_id: str = "HalfCheetah-v5",
        reward_dt: float = 0.05,
        ensemble_size: int = 5,
        hidden_dim: int = 200,
        n_layers: int = 3,
        dynamics_lr: float = 1e-3,
        dynamics_fit_steps: int = 100,
        dynamics_batch_size: int = 256,
        weight_decay: float = 1e-4,
        plan_horizon: int = 25,
        cem_population: int = 400,
        cem_elite_frac: float = 0.1,
        cem_iterations: int = 5,
        cem_alpha: float = 0.1,
        risk_beta: float = 0.0,
        n_particles: int = 20,
        ts_mode: str = "tsinf",
        num_warmup_steps: int = 1000,
        gamma: float = 0.99,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        del gamma  # PETS does not use a discount factor (planning is finite-horizon)

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.env_id = env_id
        self.reward_dt = float(reward_dt)
        self.ensemble_size = int(ensemble_size)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dynamics_lr = float(dynamics_lr)
        self.dynamics_fit_steps = int(dynamics_fit_steps)
        self.dynamics_batch_size = int(dynamics_batch_size)
        self.weight_decay = float(weight_decay)
        self.plan_horizon = int(plan_horizon)
        self.cem_population = int(cem_population)
        self.cem_elite_frac = float(cem_elite_frac)
        self.cem_iterations = int(cem_iterations)
        self.cem_alpha = float(cem_alpha)
        self.risk_beta = float(risk_beta)
        self.n_particles = int(n_particles)
        self.ts_mode = ts_mode
        self.num_warmup_steps = int(num_warmup_steps)
        self.seed = int(seed)

        self.device = resolve_device(device)
        torch.manual_seed(seed)

        # --- Action bounds (torch tensors on target device) ---
        self.action_low_t = torch.tensor(
            np.asarray(action_low, dtype=np.float32).flatten(),
            dtype=torch.float32, device=self.device,
        )
        self.action_high_t = torch.tensor(
            np.asarray(action_high, dtype=np.float32).flatten(),
            dtype=torch.float32, device=self.device,
        )

        # --- Dynamics ensemble ---
        input_dim = obs_dim + action_dim
        self.ensemble = ProbabilisticEnsemble(
            input_dim=input_dim,
            output_dim=obs_dim,
            ensemble_size=ensemble_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(self.device)

        # --- CEM planner ---
        self.planner = CEMPlanner(
            action_dim=action_dim,
            horizon=plan_horizon,
            population=cem_population,
            elite_frac=cem_elite_frac,
            iterations=cem_iterations,
            alpha=cem_alpha,
            action_low=self.action_low_t,
            action_high=self.action_high_t,
            risk_beta=risk_beta,
        )

        # --- Reward function (partial with fixed dt) ---
        reward_fn_raw = get_reward_fn(env_id)
        self._reward_fn: Callable[..., Tensor] = partial(
            reward_fn_raw, dt=reward_dt
        )

        # --- Transition buffer ---
        self.buffer = TransitionBuffer(obs_dim=obs_dim, act_dim=action_dim)

        # Warm-start mean for CEM (reset each episode)
        self._prev_mean: Tensor | None = None

        # Track whether the model has been fitted at least once
        self._fitted = False

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: Any, *, deterministic: bool = False) -> np.ndarray:
        """Select an action given the current observation.

        During warm-up (buffer too small), returns a uniform random action.
        After warm-up, uses CEM planning over the ensemble with action-sequence
        warm-starting from the previous step.

        Parameters
        ----------
        obs:
            Current observation (numpy array or list).
        deterministic:
            Ignored (CEM is deterministic in mean; warm-up is always random).

        Returns
        -------
        np.ndarray
            Action array of dtype float32, shape [action_dim], within bounds.
        """
        if len(self.buffer) < self.num_warmup_steps or not self._fitted:
            # Warm-up: uniform random action within bounds
            low_np = self.action_low_t.cpu().numpy()
            high_np = self.action_high_t.cpu().numpy()
            return np.random.uniform(low_np, high_np).astype(np.float32)

        state_t = torch.tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32,
            device=self.device,
        )

        action_np, new_mean = self.planner.plan(
            state_t,
            self.ensemble,
            self._reward_fn,
            n_particles=self.n_particles,
            ts_mode=self.ts_mode,
            prev_mean=self._prev_mean,
        )
        # Warm-start: shift mean left by one step for the next call
        self._prev_mean = torch.cat(
            [new_mean[1:], new_mean[-1:]], dim=0
        )

        return action_np

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Push (obs, action, next_obs) to the replay buffer.

        The ``reward`` and ``done`` signals are not used by the PETS dynamics
        model (the reward function is hand-coded) but are accepted for API
        compatibility.
        """
        del reward, done
        self.buffer.push(obs, action, next_obs)

    # ------------------------------------------------------------------
    # Episode lifecycle hooks
    # ------------------------------------------------------------------

    def episode_ended(self) -> None:
        """Reset the CEM warm-start mean at the end of each episode."""
        self._prev_mean = None

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Refit the dynamics ensemble on all buffered transitions.

        Calls ``fit_normalizer`` on the full dataset then trains each member
        with bootstrap resampling.

        Returns
        -------
        dict
            ``{"dynamics_nll": float, "ensemble_disagreement": float,
               "buffer_size": float}`` — all finite floats.
        """
        del kwargs
        X, Y = self.buffer.get_tensors()
        X = X.to(self.device)
        Y = Y.to(self.device)

        nll = self.ensemble.fit(
            X, Y,
            steps=self.dynamics_fit_steps,
            batch_size=self.dynamics_batch_size,
            lr=self.dynamics_lr,
            weight_decay=self.weight_decay,
        )
        self._fitted = True

        # Epistemic uncertainty over training data
        with torch.no_grad():
            disagreement = self.ensemble.disagreement(X)

        nll = float(nll) if math.isfinite(nll) else 0.0
        disagreement = float(disagreement) if math.isfinite(disagreement) else 0.0

        return {
            "dynamics_nll": nll,
            "ensemble_disagreement": disagreement,
            "buffer_size": float(len(self.buffer)),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist agent state to *path* and return the written path.

        Saves the ensemble (state dict + normalization buffers), buffer
        contents, and key hyperparameters.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "ensemble": self.ensemble.state_dict(),
            "buffer": self.buffer.state_dict(),
            "hyperparams": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "env_id": self.env_id,
                "reward_dt": self.reward_dt,
                "ensemble_size": self.ensemble_size,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "dynamics_lr": self.dynamics_lr,
                "dynamics_fit_steps": self.dynamics_fit_steps,
                "dynamics_batch_size": self.dynamics_batch_size,
                "weight_decay": self.weight_decay,
                "plan_horizon": self.plan_horizon,
                "cem_population": self.cem_population,
                "cem_elite_frac": self.cem_elite_frac,
                "cem_iterations": self.cem_iterations,
                "cem_alpha": self.cem_alpha,
                "risk_beta": self.risk_beta,
                "n_particles": self.n_particles,
                "ts_mode": self.ts_mode,
                "num_warmup_steps": self.num_warmup_steps,
                "seed": self.seed,
            },
            "action_low": self.action_low_t.cpu(),
            "action_high": self.action_high_t.cpu(),
        }
        torch.save(payload, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "PetsAgent":
        """Restore a ``PetsAgent`` from a checkpoint.

        The caller must pass at least the environment-specific kwargs that are
        not stored in the checkpoint (``action_low``, ``action_high`` are
        re-loaded from the checkpoint itself; ``obs_dim`` and ``action_dim``
        are read from hyperparams).  Any ``**kwargs`` override the stored
        hyperparams.
        """
        payload = torch.load(Path(path), weights_only=False)
        hp = payload.get("hyperparams", {})
        hp.update(kwargs)

        # Restore action bounds from checkpoint if not overridden
        if "action_low" not in hp and "action_low" in payload:
            hp["action_low"] = payload["action_low"].numpy()
        if "action_high" not in hp and "action_high" in payload:
            hp["action_high"] = payload["action_high"].numpy()

        agent = cls(**hp)
        agent.ensemble.load_state_dict(payload["ensemble"])
        agent.buffer.load_state_dict(payload.get("buffer", {}))
        agent._fitted = len(agent.buffer) > 0
        return agent
