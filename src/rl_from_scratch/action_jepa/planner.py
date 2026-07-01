"""Latent-space CEM planner for Action-JEPA."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


ObjectiveFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
PredictorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LatentCEMPlanner:
    """Plan a continuous action sequence directly in latent space."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        population: int,
        num_elites: int,
        iterations: int,
        alpha: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        seed: int = 0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.population = int(population)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.action_low = action_low.detach().clone().float()
        self.action_high = action_high.detach().clone().float()
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self._init_std = ((self.action_high - self.action_low).clamp_min(1e-6) / 2.0)

    def get_rng_state(self) -> torch.Tensor:
        return self._generator.get_state()

    def set_rng_state(self, state: torch.Tensor) -> None:
        self._generator.set_state(state.cpu())

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,
        predictor: PredictorFn,
        *,
        objective: ObjectiveFn,
        prev_mean: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        device = z0.device
        low = self.action_low.to(device)
        high = self.action_high.to(device)

        if prev_mean is None:
            mean = torch.zeros(self.horizon, self.action_dim, device=device)
        else:
            mean = prev_mean.to(device).clone()
        std = self._init_std.to(device).unsqueeze(0).expand(self.horizon, self.action_dim).clone()

        for _ in range(self.iterations):
            noise = torch.randn(
                self.population,
                self.horizon,
                self.action_dim,
                generator=self._generator,
            ).to(device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(low.view(1, 1, -1), high.view(1, 1, -1))

            latents = []
            latent = z0.unsqueeze(0).expand(self.population, -1)
            for step in range(self.horizon):
                latent = predictor(latent, actions[:, step, :])
                latents.append(latent)
            rollout_latents = torch.stack(latents, dim=1)

            scores = objective(rollout_latents, actions, z0)
            elite_idx = scores.topk(self.num_elites).indices
            elites = actions[elite_idx]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, correction=False).clamp_min(1e-4)
            mean = (1.0 - self.alpha) * mean + self.alpha * elite_mean
            std = (1.0 - self.alpha) * std + self.alpha * elite_std

        mean = mean.clamp(low.view(1, -1), high.view(1, -1))
        return mean[0].detach().cpu().numpy().astype(np.float32), mean.detach().cpu()


def goal_objective(z_goal: torch.Tensor) -> ObjectiveFn:
    """Score a rollout by how closely it reaches the latent goal."""
    z_goal = z_goal.detach()

    def _objective(
        rollout_latents: torch.Tensor,
        actions: torch.Tensor,
        z0: torch.Tensor,
    ) -> torch.Tensor:
        del z0
        del actions
        distances = (rollout_latents - z_goal.view(1, 1, -1)).pow(2).sum(dim=-1)
        return -distances.sum(dim=1)

    return _objective


def reward_objective(
    reward_head: torch.nn.Module,
    continuation_head: torch.nn.Module,
    gamma: float,
) -> ObjectiveFn:
    """Score a rollout by predicted discounted reward under continuation."""

    def _objective(
        rollout_latents: torch.Tensor,
        actions: torch.Tensor,
        z0: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros(rollout_latents.shape[0], device=rollout_latents.device)
        survival = torch.ones(rollout_latents.shape[0], device=rollout_latents.device)
        discount = 1.0
        # Reward and continuation are functions of the latent before action a_t.
        prev_latents = torch.cat(
            [
                z0.view(1, -1).expand(rollout_latents.shape[0], -1).unsqueeze(1),
                rollout_latents[:, :-1, :],
            ],
            dim=1,
        )
        for step in range(rollout_latents.shape[1]):
            prev_latent = prev_latents[:, step, :]
            reward = reward_head(prev_latent, actions[:, step, :])
            continuation = torch.sigmoid(continuation_head(prev_latent, actions[:, step, :]))
            # Compounding survival: weight r_t by the probability the rollout is
            # still alive at step t (product of prior continuations).
            total = total + discount * survival * reward
            survival = survival * continuation
            discount *= gamma
        return total

    return _objective
