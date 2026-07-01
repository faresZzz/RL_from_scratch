"""Autonomous TRPO and PPO agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.diagnostics import ActionDiagnosticsMixin
from rl_from_scratch.core.normalization import ObservationNormalizer
from rl_from_scratch.core.utils import resolve_device
from rl_from_scratch.trust_region.buffer import TrustRegionRolloutBuffer
from rl_from_scratch.trust_region.network import CriticNetwork, GaussianActor


class TrustRegionAgent(ActionDiagnosticsMixin, BaseAgent):
    """Shared trust-region agent plumbing kept local to this package."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        normalize_observations: bool = False,
        obs_norm_epsilon: float = 1e-8,
        obs_norm_clip: float = 10.0,
        device: str = "auto",
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)
        self.buffer = TrustRegionRolloutBuffer(n_steps, obs_dim, action_dim)
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.obs_normalizer = (
            ObservationNormalizer(obs_dim, epsilon=obs_norm_epsilon, clip=obs_norm_clip)
            if normalize_observations
            else None
        )

        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._reset_action_diagnostics()

    def select_action(self, observation: Any, *, deterministic: bool = False) -> np.ndarray:
        obs_t = self._to_tensor(observation)
        with torch.no_grad():
            dist = self.actor.get_distribution(obs_t)
            action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(obs_t)
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        return action.squeeze(0).cpu().numpy()

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        self.buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            bool(done),
            self._last_log_prob,
            self._last_value,
        )

    def learn_step(self, *, next_value: float = 0.0, **kwargs: Any) -> dict[str, float]:
        # GAE turns the rollout into low-variance advantages and bootstrapped returns.
        returns, advantages = self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        batch = self.buffer.get_batch(device=self.device)
        result = self._update(
            advantages.to(self.device),
            returns.to(self.device),
            batch,
        )
        result.update(self._consume_action_diagnostics(default_actions=batch["actions"]))
        self.buffer.reset()
        return result

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        extra_state = self._extra_checkpoint_state()
        checkpoint.update(extra_state)
        if self.obs_normalizer is not None:
            checkpoint["obs_normalizer"] = self.obs_normalizer.to_dict()
        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        obs_dim: int,
        action_dim: int,
        **kwargs: Any,
    ) -> TrustRegionAgent:
        agent = cls(obs_dim, action_dim, **kwargs)
        checkpoint = torch.load(Path(path), weights_only=True, map_location="cpu")
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        if "obs_normalizer" in checkpoint:
            agent.obs_normalizer = ObservationNormalizer.from_dict(checkpoint["obs_normalizer"])
        agent._load_extra_checkpoint_state(checkpoint)
        return agent

    def _update(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        raise NotImplementedError

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {}

    def _load_extra_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        del checkpoint

    def record_action_diagnostics(self, raw_action: Any, clipped_action: Any) -> None:
        raw_arr = np.asarray(raw_action, dtype=np.float32)
        clipped_arr = np.asarray(clipped_action, dtype=np.float32)
        self._action_abs_sum += float(np.mean(np.abs(clipped_arr)))
        self._action_clip_fraction_sum += float(
            np.mean(np.abs(raw_arr - clipped_arr) > 1e-6)
        )
        self._action_metric_count += 1

    def _normalized_advantages(self, advantages: Tensor) -> tuple[Tensor, float, float]:
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.detach(), adv_mean, adv_std

    def _policy_diagnostics(
        self,
        *,
        returns: Tensor,
        predicted_values: Tensor,
    ) -> dict[str, float]:
        returns_var = returns.var(unbiased=False)
        if returns_var > 1e-8:
            explained_variance = 1.0 - (
                (returns - predicted_values).var(unbiased=False) / (returns_var + 1e-8)
            )
        else:
            explained_variance = torch.tensor(0.0, device=returns.device)
        log_std = self.actor.log_std.detach()
        return {
            "explained_variance": float(explained_variance.item()),
            "log_std_mean": float(log_std.mean().item()),
            "log_std_min": float(log_std.min().item()),
            "log_std_max": float(log_std.max().item()),
        }


class TRPOAgent(TrustRegionAgent):
    """TRPO agent with local natural-gradient update math."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        max_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        value_train_iters: int = 80,
        value_lr: float = 1e-3,
        normalize_observations: bool = False,
        obs_norm_epsilon: float = 1e-8,
        obs_norm_clip: float = 10.0,
        device: str = "auto",
    ) -> None:
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            n_steps=n_steps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            normalize_observations=normalize_observations,
            obs_norm_epsilon=obs_norm_epsilon,
            obs_norm_clip=obs_norm_clip,
            device=device,
        )
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.value_train_iters = value_train_iters
        self.value_lr = value_lr
        self.value_optimizer = Adam(self.critic.parameters(), lr=value_lr)

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {"value_optimizer": self.value_optimizer.state_dict()}

    def _load_extra_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        if "value_optimizer" in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

    def _update(
        self,
        advantages: Tensor,
        returns: Tensor,
        batch: dict[str, Tensor],
    ) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"].detach()
        advantages, adv_mean, adv_std = self._normalized_advantages(advantages)

        with torch.no_grad():
            old_dist = self.actor.get_distribution(obs)
            entropy = old_dist.entropy().sum(dim=-1).mean()

        old_params = self._get_flat_params(self.actor).clone()

        def surrogate_objective() -> Tensor:
            dist = self.actor.get_distribution(obs)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            return (ratio * advantages).mean()

        actor_params = list(self.actor.parameters())
        surrogate = surrogate_objective()
        gradient = self._flat_grad(surrogate, actor_params)

        # Conjugate gradient approximately solves F x = g without materializing the Fisher matrix.
        step_direction = self._conjugate_gradient(
            lambda vector: self._fisher_vector_product(vector, obs, old_dist),
            gradient,
            n_steps=self.cg_iters,
        )
        step_curvature = torch.dot(
            step_direction,
            self._fisher_vector_product(step_direction, obs, old_dist),
        )
        # TRPO rescales the natural-gradient step to stay inside the KL trust region.
        step_scale = torch.sqrt(2.0 * self.max_kl / (step_curvature + 1e-8))
        full_step = step_scale * step_direction

        objective_before = float(surrogate_objective().item())
        accepted = 0.0
        step_fraction = 0.0
        actual_kl = 0.0
        policy_loss = -objective_before

        # Backtracking line search keeps the update only if the surrogate improves under the KL cap.
        for iteration in range(self.backtrack_iters):
            fraction = self.backtrack_coeff ** iteration
            self._set_flat_params(self.actor, old_params + fraction * full_step)
            with torch.no_grad():
                candidate_dist = self.actor.get_distribution(obs)
                candidate_objective = float(surrogate_objective().item())
                kl = torch.distributions.kl_divergence(old_dist, candidate_dist)
                candidate_kl = float(kl.sum(dim=-1).mean().item())

            if candidate_objective > objective_before and candidate_kl <= self.max_kl:
                accepted = 1.0
                step_fraction = float(fraction)
                actual_kl = candidate_kl
                policy_loss = -candidate_objective
                break
        else:
            self._set_flat_params(self.actor, old_params)

        value_loss_accum = 0.0
        grad_norm_accum = 0.0
        for _ in range(self.value_train_iters):
            predicted_values = self.critic(obs)
            value_loss = nn.functional.mse_loss(predicted_values, returns.detach())
            self.value_optimizer.zero_grad()
            value_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            value_loss_accum += float(value_loss.item())
            grad_norm_accum += float(grad_norm.item())

        with torch.no_grad():
            final_values = self.critic(obs)

        result = {
            "policy_loss": float(policy_loss),
            "value_loss": value_loss_accum / self.value_train_iters,
            "kl": float(actual_kl),
            "actual_kl": float(actual_kl),
            "entropy": float(entropy.item()),
            "line_search_accept": accepted,
            "line_search_step_fraction": step_fraction,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "grad_norm": grad_norm_accum / self.value_train_iters,
        }
        result.update(self._policy_diagnostics(returns=returns, predicted_values=final_values))
        return result

    def _fisher_vector_product(
        self,
        vector: Tensor,
        obs: Tensor,
        old_dist: torch.distributions.Normal,
    ) -> Tensor:
        actor_params = list(self.actor.parameters())
        new_dist = self.actor.get_distribution(obs)
        kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1).mean()
        grads = torch.autograd.grad(kl, actor_params, create_graph=True)
        flat_grad = torch.cat([grad.reshape(-1) for grad in grads])
        grad_vector_dot = torch.dot(flat_grad, vector.detach())
        second_grads = torch.autograd.grad(grad_vector_dot, actor_params)
        flat_second_grads = torch.cat([grad.reshape(-1) for grad in second_grads])
        return flat_second_grads + self.cg_damping * vector.detach()

    def _conjugate_gradient(
        self,
        av_fn: Any,
        rhs: Tensor,
        *,
        n_steps: int,
        residual_tol: float = 1e-10,
    ) -> Tensor:
        solution = torch.zeros_like(rhs)
        residual = rhs.clone().detach()
        direction = residual.clone()
        residual_dot = torch.dot(residual, residual)

        for _ in range(n_steps):
            av_direction = av_fn(direction)
            alpha = residual_dot / (torch.dot(direction, av_direction) + 1e-8)
            solution = solution + alpha * direction
            residual = residual - alpha * av_direction
            new_residual_dot = torch.dot(residual, residual)
            if new_residual_dot < residual_tol:
                break
            beta = new_residual_dot / (residual_dot + 1e-8)
            direction = residual + beta * direction
            residual_dot = new_residual_dot

        return solution

    def _get_flat_params(self, model: nn.Module) -> Tensor:
        return torch.cat([parameter.data.reshape(-1) for parameter in model.parameters()])

    def _set_flat_params(self, model: nn.Module, flat_params: Tensor) -> None:
        offset = 0
        for parameter in model.parameters():
            numel = parameter.numel()
            parameter.data.copy_(flat_params[offset : offset + numel].reshape(parameter.shape))
            offset += numel

    def _flat_grad(self, loss: Tensor, params: list[Tensor]) -> Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        return torch.cat([grad.reshape(-1) for grad in grads]).detach()


class PPOAgent(TrustRegionAgent):
    """PPO agent with clipped-surrogate local update math."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        target_kl: float = 0.01,
        normalize_observations: bool = False,
        obs_norm_epsilon: float = 1e-8,
        obs_norm_clip: float = 10.0,
        device: str = "auto",
    ) -> None:
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            n_steps=n_steps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            normalize_observations=normalize_observations,
            obs_norm_epsilon=obs_norm_epsilon,
            obs_norm_clip=obs_norm_clip,
            device=device,
        )
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl

    def _update(
        self,
        advantages: Tensor,
        returns: Tensor,
        batch: dict[str, Tensor],
    ) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"].detach()
        advantages, adv_mean, adv_std = self._normalized_advantages(advantages)
        returns = returns.detach()

        indices = np.arange(obs.shape[0])
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        total_ratio_mean = 0.0
        total_ratio_std = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            if num_updates > 0 and total_kl / num_updates > self.target_kl:
                break

            for start in range(0, obs.shape[0], self.batch_size):
                stop = min(start + self.batch_size, obs.shape[0])
                minibatch_idx = indices[start:stop]

                mb_obs = obs[minibatch_idx]
                mb_actions = actions[minibatch_idx]
                mb_old_log_probs = old_log_probs[minibatch_idx]
                mb_advantages = advantages[minibatch_idx]
                mb_returns = returns[minibatch_idx]

                dist = self.actor.get_distribution(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                predicted_values = self.critic(mb_obs)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                # PPO clips the importance ratio so overly large policy moves stop improving the objective.
                unclipped_surrogate = ratio * mb_advantages
                clipped_surrogate = ratio.clamp(
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * mb_advantages
                policy_loss = -torch.min(unclipped_surrogate, clipped_surrogate).mean()
                value_loss = nn.functional.mse_loss(predicted_values, mb_returns)
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (new_log_probs - mb_old_log_probs)).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_kl += float(approx_kl.item())
                total_clip_fraction += float(clip_fraction.item())
                total_ratio_mean += float(ratio.mean().item())
                total_ratio_std += float(ratio.std(unbiased=False).item())
                total_grad_norm += float(grad_norm.item())
                num_updates += 1

        if num_updates == 0:
            num_updates = 1

        with torch.no_grad():
            final_values = self.critic(obs)

        result = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl": total_kl / num_updates,
            "approx_kl": total_kl / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "ratio_mean": total_ratio_mean / num_updates,
            "ratio_std": total_ratio_std / num_updates,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "grad_norm": total_grad_norm / num_updates,
        }
        result.update(self._policy_diagnostics(returns=returns, predicted_values=final_values))
        return result
