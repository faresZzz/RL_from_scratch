"""Cross-Entropy Method (CEM) planner for PETS.

The planner implements the trajectory-sampling CEM variant from Chua et al. 2018
(Sec. 3.2).  At each planning step:

1. Maintain a belief over action sequences as a diagonal Gaussian
   N(mu, sigma^2) over [horizon, action_dim]-shaped variables.

2. For ``cem_iterations`` refinement steps:
   a. Sample ``cem_population`` action sequences from the current belief.
   b. Evaluate each sequence by rolling the ensemble forward for ``horizon``
      steps with ``n_particles`` particles.
   c. Select the top ``elite_frac * population`` elites by cumulative reward.
   d. Update the belief with a momentum-weighted update:
          mu  <- (1 - alpha) * mu  + alpha * mean(elites)
          std <- (1 - alpha) * std + alpha * std(elites)

3. Return the first action of the optimal mean sequence and the full mean
   for warm-starting the next step.

Trajectory Sampling modes
-------------------------
TS∞ (ts_mode="tsinf"): each of the ``n_particles`` particles is permanently
    assigned to one ensemble member at the start of the horizon and uses
    that member for all ``horizon`` steps.

TS1 (ts_mode="ts1"): at each step, every particle re-samples a member index
    uniformly at random (independent draws).
"""

from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from rl_from_scratch.pets.dynamics import ProbabilisticEnsemble


class CEMPlanner:
    """Action planner based on the Cross-Entropy Method with ensemble propagation.

    Parameters
    ----------
    action_dim:
        Dimensionality of the action space.
    horizon:
        Planning horizon (number of steps per rollout).
    population:
        Number of candidate action sequences sampled per CEM iteration.
    elite_frac:
        Fraction of top candidates used to update the distribution.
    iterations:
        Number of CEM refinement iterations.
    alpha:
        CEM momentum weight; higher values trust elites more aggressively.
    risk_beta:
        Penalty applied to particle-return dispersion.  ``0`` recovers the
        standard PETS objective (mean return); positive values prefer plans
        that remain good across particles instead of relying on one optimistic
        model sample.
    action_low:
        Per-dimension action lower bounds, shape [action_dim].
    action_high:
        Per-dimension action upper bounds, shape [action_dim].
    """

    _MIN_STD: float = 1e-5  # floor on action sequence std to avoid collapse

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        population: int,
        elite_frac: float,
        iterations: int,
        alpha: float,
        action_low: Tensor,
        action_high: Tensor,
        risk_beta: float = 0.0,
    ) -> None:
        self.action_dim = action_dim
        self.horizon = horizon
        self.population = population
        self.elite_frac = elite_frac
        self.iterations = iterations
        self.alpha = alpha
        self.risk_beta = float(risk_beta)

        self.action_low = action_low    # [action_dim]
        self.action_high = action_high  # [action_dim]
        self.n_elites = max(1, math.ceil(elite_frac * population))

        # Initial std: half the action range
        action_range = (action_high - action_low).clamp_min(1e-6)
        self._init_std = (action_range / 2.0)  # [action_dim]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        state: Tensor,
        ensemble: "ProbabilisticEnsemble",
        reward_fn: Callable[..., Tensor],
        *,
        n_particles: int,
        ts_mode: str,
        prev_mean: Tensor | None = None,
    ) -> tuple[np.ndarray, Tensor]:
        """Plan from the current state and return the first action.

        Parameters
        ----------
        state:
            Current environment state, shape [obs_dim].
        ensemble:
            Fitted probabilistic ensemble dynamics model.
        reward_fn:
            Reward function with signature
            ``(obs, act, next_obs, ...) -> Tensor`` where leading dims are
            ``[population * n_particles]``.
        n_particles:
            Number of particles per candidate action sequence.
        ts_mode:
            ``"ts1"`` or ``"tsinf"`` trajectory sampling mode.
        prev_mean:
            Warm-start mean from previous step, shape [horizon, action_dim].
            If None, initialises to zeros.

        Returns
        -------
        first_action:
            The first action in the optimal sequence, as a numpy float32 array
            of shape [action_dim].
        new_mean:
            Updated CEM mean for the next planning step, shape
            [horizon, action_dim].
        """
        device = state.device
        H, A = self.horizon, self.action_dim

        # Initialise CEM distribution
        if prev_mean is not None:
            mu = prev_mean.clone().to(device)
        else:
            mu = torch.zeros(H, A, dtype=torch.float32, device=device)

        std = self._init_std.to(device).unsqueeze(0).expand(H, A).clone()  # [H, A]

        low = self.action_low.to(device)   # [A]
        high = self.action_high.to(device) # [A]

        for _ in range(self.iterations):
            # --- Sample population action sequences ---
            # shape [population, H, A]
            noise = torch.randn(self.population, H, A, device=device)
            acts = (mu.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                low.view(1, 1, A), high.view(1, 1, A)
            )  # [P, H, A]

            # --- Evaluate returns ---
            returns = self._evaluate_sequences(
                acts, state, ensemble, reward_fn,
                n_particles=n_particles, ts_mode=ts_mode,
            )  # [P]

            # --- Select elites ---
            elite_idx = returns.topk(self.n_elites).indices
            elites = acts[elite_idx]  # [n_elites, H, A]

            elite_mean = elites.mean(0)   # [H, A]
            elite_std = elites.std(0).clamp_min(self._MIN_STD)  # [H, A]

            # --- Momentum update ---
            mu = (1.0 - self.alpha) * mu + self.alpha * elite_mean
            std = (1.0 - self.alpha) * std + self.alpha * elite_std

        # Clamp final mean to action bounds and extract first action
        mu = mu.clamp(low.view(1, A), high.view(1, A))
        first_action = mu[0].detach().cpu().numpy().astype(np.float32)
        return first_action, mu.detach()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_sequences(
        self,
        acts: Tensor,
        state: Tensor,
        ensemble: "ProbabilisticEnsemble",
        reward_fn: Callable[..., Tensor],
        *,
        n_particles: int,
        ts_mode: str,
    ) -> Tensor:
        """Roll out candidate sequences and return cumulative rewards.

        Parameters
        ----------
        acts:
            Candidate action sequences, shape [P, H, A].
        state:
            Starting state, shape [obs_dim].
        ensemble:
            Fitted ensemble for dynamics propagation.
        reward_fn:
            Reward function.
        n_particles:
            Number of particles per candidate.
        ts_mode:
            ``"ts1"`` or ``"tsinf"``.

        Returns
        -------
        Tensor
            Cumulative rewards per candidate, shape [P].
        """
        P, H, A = acts.shape
        E = ensemble.ensemble_size
        obs_dim = state.shape[0]
        device = state.device
        B = P * n_particles  # total simulation batch size

        # Tile the start state for all (candidate, particle) pairs
        states = state.unsqueeze(0).expand(B, obs_dim).clone()  # [B, obs_dim]

        # Assign member indices
        if ts_mode == "tsinf":
            # Fixed member per particle, cycling across ensemble members
            particle_member = torch.arange(B, device=device) % E  # [B]
        elif ts_mode == "ts1":
            particle_member = None  # will be sampled each step
        else:
            raise ValueError(f"Unknown ts_mode: '{ts_mode}'")

        # Expand acts to match particles: [P, H, A] → [P*n_particles, H, A]
        acts_expanded = acts.repeat_interleave(n_particles, dim=0)  # [B, H, A]

        cumulative_reward = torch.zeros(B, device=device)

        for t in range(H):
            action_t = acts_expanded[:, t, :]   # [B, A]

            if ts_mode == "ts1":
                member_idx = torch.randint(0, E, (B,), device=device)
            else:
                assert particle_member is not None
                member_idx = particle_member

            next_states = ensemble.propagate(states, action_t, member_idx)  # [B, obs_dim]

            rewards_t = reward_fn(states, action_t, next_states)  # [B]
            cumulative_reward = cumulative_reward + rewards_t
            states = next_states

        # Standard PETS scores by mean particle return.  The optional
        # risk-sensitive variant subtracts a dispersion penalty, which is useful
        # when CEM otherwise selects plans that only look good for a few
        # optimistic particles but fail under other plausible ensemble members.
        particle_returns = cumulative_reward.view(P, n_particles)  # [P, n_particles]
        mean_return = particle_returns.mean(dim=1)  # [P]
        if self.risk_beta <= 0.0:
            return mean_return
        return mean_return - self.risk_beta * particle_returns.std(dim=1, unbiased=False)
