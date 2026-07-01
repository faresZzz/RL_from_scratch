"""Action-JEPA agent: latent prediction, collapse prevention, and planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_from_scratch.action_jepa.buffer import SequenceBuffer
from rl_from_scratch.action_jepa.networks import (
    ContinuationHead,
    Encoder,
    MaskedContextPredictor,
    Predictor,
    RewardHead,
    covariance_loss,
    ema_update,
    latent_collapse_metric,
    variance_loss,
)
from rl_from_scratch.action_jepa.planner import LatentCEMPlanner, goal_objective, reward_objective
from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device


def _safe_float(value: torch.Tensor | float) -> float:
    scalar = float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value)
    return scalar if math.isfinite(scalar) else 0.0


def _effective_rank(latents: torch.Tensor, eps: float = 1e-8) -> float:
    centered = latents - latents.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return _safe_float(entropy.exp())


class ActionJepaAgent(BaseAgent):
    """World-model agent trained with a JEPA latent-prediction objective."""

    def __init__(  # noqa: PLR0913
        self,
        obs_dim: int,
        action_dim: int,
        *,
        action_low: Any,
        action_high: Any,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        training_regime: str = "joint",
        freeze_encoder_after_pretrain: bool = False,
        random_frozen_encoder: bool = False,
        mask_fraction: float = 1.0 / 3.0,
        mask_noise_std: float = 0.01,
        representation_val_every: int = 10,
        ema_tau: float = 0.99,
        predictor_delta: bool = True,
        rollout_len: int = 5,
        reward_loss_weight: float = 1.0,
        continuation_loss_weight: float = 1.0,
        variance_loss_weight: float = 1.0,
        covariance_loss_weight: float = 0.04,
        target_std: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 10.0,
        batch_size: int = 128,
        buffer_capacity: int = 200_000,
        learning_starts: int = 2_000,
        num_warmup_steps: int = 2_000,
        plan_mode: str = "goal",
        goal_obs: list[float] | None = None,
        plan_horizon: int = 12,
        cem_population: int = 256,
        cem_num_elites: int = 32,
        cem_iterations: int = 4,
        cem_alpha: float = 0.1,
        gamma: float = 0.99,
        seed: int = 0,
        device: str = "auto",
        **_: Any,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_layers = int(encoder_layers)
        self.training_regime = str(training_regime)
        self.freeze_encoder_after_pretrain = bool(freeze_encoder_after_pretrain)
        self.random_frozen_encoder = bool(random_frozen_encoder)
        self.mask_fraction = float(mask_fraction)
        self.mask_noise_std = float(mask_noise_std)
        self.representation_val_every = int(representation_val_every)
        self.ema_tau = float(ema_tau)
        self.predictor_delta = bool(predictor_delta)
        self.rollout_len = int(rollout_len)
        self.reward_loss_weight = float(reward_loss_weight)
        self.continuation_loss_weight = float(continuation_loss_weight)
        self.variance_loss_weight = float(variance_loss_weight)
        self.covariance_loss_weight = float(covariance_loss_weight)
        self.target_std = float(target_std)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.batch_size = int(batch_size)
        self.buffer_capacity = int(buffer_capacity)
        self.learning_starts = int(learning_starts)
        self.num_warmup_steps = int(num_warmup_steps)
        self.plan_mode = plan_mode
        self.goal_obs = list(goal_obs) if goal_obs is not None else None
        self.plan_horizon = int(plan_horizon)
        self.cem_population = int(cem_population)
        self.cem_num_elites = int(cem_num_elites)
        self.cem_iterations = int(cem_iterations)
        self.cem_alpha = float(cem_alpha)
        self.gamma = float(gamma)
        self.seed = int(seed)

        self.device = torch.device(resolve_device(device))
        self._action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self._action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self._rng = np.random.default_rng(seed)

        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim, encoder_layers).to(self.device)
        self.target_encoder = Encoder(obs_dim, latent_dim, hidden_dim, encoder_layers).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = Predictor(latent_dim, action_dim, hidden_dim, delta=predictor_delta).to(self.device)
        self.mask_predictor = MaskedContextPredictor(latent_dim, obs_dim, hidden_dim).to(self.device)
        self.reward_head = RewardHead(latent_dim, action_dim, hidden_dim).to(self.device)
        self.cont_head = ContinuationHead(latent_dim, action_dim, hidden_dim).to(self.device)

        self.encoder_frozen = False
        self.optimizer = self._make_action_optimizer(include_encoder=True)
        self.representation_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.mask_predictor.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        if self.random_frozen_encoder:
            self.freeze_encoder()

        self.buffer = SequenceBuffer(buffer_capacity, seed=seed)
        self._collected_steps = 0
        self.planner = LatentCEMPlanner(
            action_dim=action_dim,
            horizon=plan_horizon,
            population=cem_population,
            num_elites=cem_num_elites,
            iterations=cem_iterations,
            alpha=cem_alpha,
            action_low=torch.tensor(self._action_low),
            action_high=torch.tensor(self._action_high),
            seed=seed,
        )
        self._prev_mean: torch.Tensor | None = None

    def _make_action_optimizer(self, *, include_encoder: bool) -> torch.optim.Optimizer:
        params: list[nn.Parameter] = []
        if include_encoder:
            params.extend(self.encoder.parameters())
        params.extend(self.predictor.parameters())
        params.extend(self.reward_head.parameters())
        params.extend(self.cont_head.parameters())
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    def freeze_encoder(self) -> None:
        """Freeze the online/target encoders and train only dynamics heads."""
        for module in (self.encoder, self.target_encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.encoder_frozen = True
        self.optimizer = self._make_action_optimizer(include_encoder=False)
        self.representation_optimizer = torch.optim.Adam(
            self.mask_predictor.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def unfreeze_encoder(self) -> None:
        """Re-enable joint encoder+dynamics training."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.train()
        self.encoder_frozen = False
        self.optimizer = self._make_action_optimizer(include_encoder=True)
        self.representation_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.mask_predictor.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def select_action(self, observation: Any, *, deterministic: bool = False) -> np.ndarray:
        if self._collected_steps < self.num_warmup_steps:
            return self._rng.uniform(self._action_low, self._action_high).astype(np.float32)

        obs_t = torch.tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            z0 = self.encoder(obs_t).squeeze(0)
            objective = self._planner_objective()
            action, new_mean = self.planner.plan(
                z0,
                self.predictor,
                objective=objective,
                prev_mean=self._prev_mean,
            )
            self._prev_mean = torch.cat([new_mean[1:], new_mean[-1:]], dim=0)
        clipped = np.clip(action, self._action_low, self._action_high)
        return clipped.astype(np.float32)

    def _planner_objective(self):
        if self.plan_mode == "goal":
            if self.goal_obs is None:
                raise ValueError("goal_obs is required in goal mode.")
            with torch.no_grad():
                goal_obs = torch.tensor(self.goal_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                z_goal = self.target_encoder(goal_obs).squeeze(0)
            return goal_objective(z_goal)
        return reward_objective(self.reward_head, self.cont_head, self.gamma)

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        self.buffer.add(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        self._collected_steps += 1

    def episode_ended(self) -> None:
        self.buffer.flush_current()
        self._prev_mean = None

    def _make_partial_view(self, target_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, obs_dim = target_obs.shape
        n_masked = min(obs_dim - 1, max(1, round(self.mask_fraction * obs_dim)))
        visible = torch.ones_like(target_obs)
        for row in range(batch_size):
            indices = torch.randperm(obs_dim, device=target_obs.device)[:n_masked]
            visible[row, indices] = 0.0
        if self.mask_noise_std > 0.0:
            noisy = target_obs + self.mask_noise_std * torch.randn_like(target_obs)
        else:
            noisy = target_obs
        return noisy * visible, visible

    def representation_step(self, *, update: bool = True) -> dict[str, float]:
        """Run one Phase-A masked JEPA update.

        The student receives the current observation and a partial/noisy view of
        the next observation.  The EMA teacher encodes the full next
        observation and is always stop-gradient.
        """
        zero = {
            "representation_prediction_loss": 0.0,
            "representation_variance_loss": 0.0,
            "representation_covariance_loss": 0.0,
            "latent_std": 0.0,
            "effective_rank": 0.0,
        }
        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return zero
        try:
            batch = self.buffer.sample(self.batch_size, 1)
        except RuntimeError:
            return zero

        context_obs = batch["obs"][:, 0].to(self.device)
        target_obs = batch["obs"][:, 1].to(self.device)
        partial_obs, visible_mask = self._make_partial_view(target_obs)

        context_z = self.encoder(context_obs)
        partial_z = self.encoder(partial_obs)
        prediction = self.mask_predictor(context_z, partial_z, visible_mask)
        with torch.no_grad():
            target_z = self.target_encoder(target_obs)

        prediction_loss = F.l1_loss(prediction, target_z)
        var_loss = variance_loss(context_z, target_std=self.target_std)
        cov_loss = covariance_loss(context_z)
        total_loss = (
            prediction_loss
            + self.variance_loss_weight * var_loss
            + self.covariance_loss_weight * cov_loss
        )

        if update:
            self.representation_optimizer.zero_grad()
            total_loss.backward()
            params = list(self.encoder.parameters()) + list(self.mask_predictor.parameters())
            nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], self.grad_clip)
            self.representation_optimizer.step()
            ema_update(self.target_encoder, self.encoder, tau=self.ema_tau)

        latent_std = latent_collapse_metric(context_z.detach())
        rank = _effective_rank(context_z.detach())
        return {
            "representation_prediction_loss": _safe_float(prediction_loss),
            "representation_variance_loss": _safe_float(var_loss),
            "representation_covariance_loss": _safe_float(cov_loss),
            "latent_std": _safe_float(latent_std),
            "effective_rank": _safe_float(rank),
        }

    def learn_step(self, **_: Any) -> dict[str, float]:
        zero = {
            "latent_prediction_loss": 0.0,
            "rollout_prediction_loss": 0.0,
            "reward_prediction_loss": 0.0,
            "continuation_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
            "total_loss": 0.0,
            "latent_std": 0.0,
            "collapse_gap": 1.0,
        }
        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return zero

        try:
            batch = self.buffer.sample(self.batch_size, self.rollout_len)
        except RuntimeError:
            return zero

        obs = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)

        flat_obs = obs.reshape(-1, self.obs_dim)
        if self.encoder_frozen:
            with torch.no_grad():
                online_latents = self.encoder(flat_obs).reshape(
                    self.batch_size,
                    self.rollout_len + 1,
                    self.latent_dim,
                )
        else:
            online_latents = self.encoder(flat_obs).reshape(
                self.batch_size,
                self.rollout_len + 1,
                self.latent_dim,
            )
        with torch.no_grad():
            target_latents = self.target_encoder(flat_obs).reshape(
                self.batch_size,
                self.rollout_len + 1,
                self.latent_dim,
            )

        latent = online_latents[:, 0, :]
        pred_latents: list[torch.Tensor] = []
        pred_rewards: list[torch.Tensor] = []
        pred_conts: list[torch.Tensor] = []

        # Roll the predictor forward for K latent steps and supervise each step.
        # Reward/continuation heads consume the ENCODER latent before action a_t
        # (z_t), matching the notebook and how the planner queries them — not the
        # rolled-out imagined latent.
        for step in range(self.rollout_len):
            action_t = actions[:, step, :]
            z_t = online_latents[:, step, :]
            pred_rewards.append(self.reward_head(z_t, action_t))
            pred_conts.append(self.cont_head(z_t, action_t))
            latent = self.predictor(latent, action_t)
            pred_latents.append(latent)

        pred_latents_t = torch.stack(pred_latents, dim=1)
        pred_rewards_t = torch.stack(pred_rewards, dim=1)
        pred_conts_t = torch.stack(pred_conts, dim=1)
        target_next = target_latents[:, 1:, :].detach()

        # One-step JEPA loss is the closest analogue to (o_t, a_t) -> z_{t+1}.
        latent_prediction_loss = F.mse_loss(pred_latents_t[:, 0, :], target_next[:, 0, :])
        # Multi-step rollout loss exposes compounding drift in latent imagination.
        rollout_prediction_loss = F.mse_loss(pred_latents_t, target_next)
        reward_prediction_loss = F.mse_loss(pred_rewards_t, rewards)
        continuation_targets = 1.0 - dones
        continuation_loss = F.binary_cross_entropy_with_logits(pred_conts_t, continuation_targets)

        # These VICReg-lite terms keep per-dim spread and reduce redundancy.
        flattened = online_latents.reshape(-1, self.latent_dim)
        var_loss = variance_loss(flattened, target_std=self.target_std)
        cov_loss = covariance_loss(flattened)

        total_loss = (
            rollout_prediction_loss
            + self.reward_loss_weight * reward_prediction_loss
            + self.continuation_loss_weight * continuation_loss
            + self.variance_loss_weight * var_loss
            + self.covariance_loss_weight * cov_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        params = (
            list(self.predictor.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters())
        )
        if not self.encoder_frozen:
            params = list(self.encoder.parameters()) + params
        nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.optimizer.step()
        if not self.encoder_frozen:
            ema_update(self.target_encoder, self.encoder, tau=self.ema_tau)

        latent_std = latent_collapse_metric(flattened.detach())

        def _safe(value: torch.Tensor | float) -> float:
            scalar = float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value)
            return scalar if math.isfinite(scalar) else 0.0

        return {
            "latent_prediction_loss": _safe(latent_prediction_loss),
            "rollout_prediction_loss": _safe(rollout_prediction_loss),
            "reward_prediction_loss": _safe(reward_prediction_loss),
            "continuation_loss": _safe(continuation_loss),
            "variance_loss": _safe(var_loss),
            "covariance_loss": _safe(cov_loss),
            "total_loss": _safe(total_loss),
            "latent_std": _safe(latent_std),
            "collapse_gap": _safe(max(0.0, self.target_std - latent_std)),
        }

    def save(self, path: Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": self.encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "mask_predictor": self.mask_predictor.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "cont_head": self.cont_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "representation_optimizer": self.representation_optimizer.state_dict(),
            "planner_rng_state": self.planner.get_rng_state(),
            "prev_mean": self._prev_mean,
            "meta": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self._action_low.tolist(),
                "action_high": self._action_high.tolist(),
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "encoder_layers": self.encoder_layers,
                "training_regime": self.training_regime,
                "freeze_encoder_after_pretrain": self.freeze_encoder_after_pretrain,
                "random_frozen_encoder": self.random_frozen_encoder,
                "mask_fraction": self.mask_fraction,
                "mask_noise_std": self.mask_noise_std,
                "representation_val_every": self.representation_val_every,
                "ema_tau": self.ema_tau,
                "predictor_delta": self.predictor_delta,
                "rollout_len": self.rollout_len,
                "reward_loss_weight": self.reward_loss_weight,
                "continuation_loss_weight": self.continuation_loss_weight,
                "variance_loss_weight": self.variance_loss_weight,
                "covariance_loss_weight": self.covariance_loss_weight,
                "target_std": self.target_std,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "grad_clip": self.grad_clip,
                "batch_size": self.batch_size,
                "buffer_capacity": self.buffer_capacity,
                "learning_starts": self.learning_starts,
                "num_warmup_steps": self.num_warmup_steps,
                "plan_mode": self.plan_mode,
                "goal_obs": self.goal_obs,
                "plan_horizon": self.plan_horizon,
                "cem_population": self.cem_population,
                "cem_num_elites": self.cem_num_elites,
                "cem_iterations": self.cem_iterations,
                "cem_alpha": self.cem_alpha,
                "gamma": self.gamma,
                "seed": self.seed,
                "collected_steps": self._collected_steps,
                "encoder_frozen": self.encoder_frozen,
            },
        }
        torch.save(payload, output_path)
        return output_path

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "ActionJepaAgent":
        payload = torch.load(Path(path), weights_only=False, map_location="cpu")
        meta = dict(payload["meta"])
        if "device" in kwargs:
            meta["device"] = kwargs["device"]
        agent = cls(**meta)
        agent.encoder.load_state_dict(payload["encoder"])
        agent.target_encoder.load_state_dict(payload["target_encoder"])
        agent.predictor.load_state_dict(payload["predictor"])
        agent.mask_predictor.load_state_dict(payload["mask_predictor"])
        agent.reward_head.load_state_dict(payload["reward_head"])
        agent.cont_head.load_state_dict(payload["cont_head"])
        if bool(meta.get("encoder_frozen", False)):
            agent.freeze_encoder()
        agent.optimizer.load_state_dict(payload["optimizer"])
        if "representation_optimizer" in payload:
            agent.representation_optimizer.load_state_dict(payload["representation_optimizer"])
        agent._prev_mean = payload.get("prev_mean")
        agent._collected_steps = int(meta.get("collected_steps", 0))
        rng_state = payload.get("planner_rng_state")
        if rng_state is not None:
            agent.planner.set_rng_state(rng_state)
        return agent
