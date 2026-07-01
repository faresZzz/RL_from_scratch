"""A2C (Advantage Actor-Critic), A2C-GAE, and A3C agents.

Designed to be extensible toward A3C: _compute_advantages() and _update()
are separate methods to allow reuse by asynchronous workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.diagnostics import ActionDiagnosticsMixin
from rl_from_scratch.actor_critic.network import GaussianActor, CriticNetwork
from rl_from_scratch.actor_critic.buffer import RolloutBuffer
from rl_from_scratch.actor_critic.optim import SharedAdam
from rl_from_scratch.core.normalization import ObservationNormalizer


class A2CAgent(ActionDiagnosticsMixin, BaseAgent):
    """Advantage Actor-Critic (A2C) agent for continuous action spaces.

    Collects n_steps transitions, computes N-step returns and advantages,
    then performs a joint update of the actor and critic via a single Adam
    optimizer.

    Designed for A3C extensibility: ``_compute_advantages()`` and ``_update()``
    are separate methods that A3C workers can reuse.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    config:
        Optional config dataclass; the hyperparameters are extracted from it.
    hidden_dim:
        Width of each hidden layer.
    lr:
        Learning rate of the Adam optimizer.
    gamma:
        Discount factor.
    n_steps:
        Number of time steps per rollout.
    entropy_coef:
        Coefficient of the entropy in the total loss (encourages exploration).
    value_coef:
        Coefficient of the value loss in the total loss.
    max_grad_norm:
        Maximum norm for gradient clipping.
    normalize_observations:
        If ``True``, normalize observations by running mean/variance.
    obs_norm_epsilon:
        Value added to the variance to avoid division by zero.
    obs_norm_clip:
        Maximum absolute value of the normalized observation.
    """

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
        normalize_observations: bool = False,
        obs_norm_epsilon: float = 1e-8,
        obs_norm_clip: float = 10.0,
        device: str = "auto",
    ) -> None:
        from rl_from_scratch.core.utils import resolve_device
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

        # Actor and critic networks
        self.actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # Joint optimizer (standard A2C — a single lr for both networks)
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        # On-policy rollout buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim)

        # Observation normalizer (optional)
        if normalize_observations:
            self.obs_normalizer: ObservationNormalizer | None = ObservationNormalizer(
                obs_dim, epsilon=obs_norm_epsilon, clip=obs_norm_clip
            )
        else:
            self.obs_normalizer = None

        # Cache for store_transition (filled by select_action)
        self._last_log_prob: float = 0.0
        self._last_value: float = 0.0
        self._reset_action_diagnostics()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> np.ndarray:
        """Choose a continuous action under the current Gaussian policy.

        In stochastic mode, sample from N(μ, σ²).
        In deterministic mode, return μ directly.

        The log-prob and the value V(s) are cached for ``store_transition``.

        Parameters
        ----------
        observation:
            Current observation from the environment.
        deterministic:
            If True, return the mean (without exploration noise).

        Returns
        -------
        np.ndarray
            Action of shape ``(action_dim,)``.
        """
        obs_t = self._to_tensor(observation)

        with torch.no_grad():
            dist = self.actor.get_distribution(obs_t)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

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
        """Record the transition in the rollout buffer.

        Uses the log-prob and value cached by ``select_action``.

        Parameters
        ----------
        obs:
            Observation of the current state.
        action:
            Action taken.
        reward:
            Reward received.
        next_obs:
            Next observation (not used directly — next_value is computed
            externally when learn_step is called).
        done:
            End-of-episode flag.
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)
        self.buffer.push(
            obs_arr,
            action_arr,
            float(reward),
            bool(done),
            self._last_log_prob,
            self._last_value,
        )

    def learn_step(self, *, next_value: float = 0.0, **kwargs: Any) -> dict[str, float]:
        """Orchestrate a full A2C update.

        1. Compute advantages and returns.
        2. Run the gradient update.
        3. Reset the buffer.

        Parameters
        ----------
        next_value:
            V(s_{T+1}) — 0.0 if the episode has terminated, otherwise the
            estimated value of the next state.

        Returns
        -------
        dict[str, float]
            ``{"policy_loss", "value_loss", "entropy", "total_loss"}``
        """
        returns, advantages = self._compute_advantages(next_value)
        batch = self.buffer.get_batch(device=self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        result = self._update(advantages, returns, batch)
        result.update(self._consume_action_diagnostics(default_actions=batch["actions"]))
        self.buffer.reset()
        return result

    def episode_ended(self) -> None:
        """No action required at the end of an episode for A2C."""

    def save(self, path: str | Path) -> Path:
        """Save the state dicts of the actor and the critic.

        If an observation normalizer is active, its statistics are included
        in the checkpoint under the ``"obs_normalizer"`` key.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
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
    ) -> A2CAgent:
        """Load an agent from a saved checkpoint.

        If the checkpoint contains an ``"obs_normalizer"`` key, the normalizer
        is restored automatically with its current statistics.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.
        obs_dim:
            Observation dimensionality (must match the checkpoint).
        action_dim:
            Action dimensionality.
        """
        agent = cls(obs_dim, action_dim, **kwargs)
        checkpoint = torch.load(Path(path), weights_only=True)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        if "obs_normalizer" in checkpoint:
            agent.obs_normalizer = ObservationNormalizer.from_dict(
                checkpoint["obs_normalizer"]
            )
        return agent

    # ------------------------------------------------------------------
    # Fork point for A3C / A2C-GAE
    # ------------------------------------------------------------------

    def _compute_advantages(
        self, next_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute N-step advantages (bootstrapped returns).

        Fork point: A2CGAEAgent overrides this method to use GAE. A3C may
        also override it to compute the returns in a distributed manner.

        Parameters
        ----------
        next_value:
            V(s_{T+1}) after the end of the rollout.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(returns, advantages)`` each of shape ``(n_steps,)``.
        """
        return self.buffer.compute_returns(next_value, self.gamma)

    def _update(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Compute the losses and perform a backpropagation step.

        Reusable by A3C workers that accumulate gradients asynchronously.

        Parameters
        ----------
        advantages:
            Advantages A_t of shape ``(n_steps,)``.
        returns:
            Returns R_t of shape ``(n_steps,)``, targets for the critic.
        batch:
            Dictionary ``{obs, actions, log_probs, values}`` as tensors.

        Returns
        -------
        dict[str, float]
            ``{"policy_loss", "value_loss", "entropy", "total_loss"}``
        """
        obs = batch["obs"]
        actions = batch["actions"]
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())

        # Advantage normalization to stabilize training
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Fresh forward pass to enable gradient computation
        dist = self.actor.get_distribution(obs)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)  # (n_steps,)
        entropy = dist.entropy().sum(dim=-1).mean()          # scalar

        new_values = self.critic(obs)  # (n_steps,)

        # Policy loss: -E[log π(a|s) * A]
        policy_loss = -(new_log_probs * advantages.detach()).mean()

        # Value loss: MSE(V(s), R)
        value_loss = torch.nn.functional.mse_loss(new_values, returns.detach())

        # Combined total loss
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
            returns_var = returns.var(unbiased=False)
            if returns_var > 1e-8:
                explained_variance = 1.0 - (
                    (returns - new_values).var(unbiased=False) / (returns_var + 1e-8)
                )
            else:
                explained_variance = torch.tensor(0.0, device=self.device)
            log_std = self.actor.log_std.detach()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "explained_variance": float(explained_variance.item()),
            "grad_norm": float(grad_norm.item()),
            "log_std_mean": float(log_std.mean().item()),
            "log_std_min": float(log_std.min().item()),
            "log_std_max": float(log_std.max().item()),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def record_action_diagnostics(self, raw_action: Any, clipped_action: Any) -> None:
        """Accumulate action diagnostics for the next update."""
        raw_arr = np.asarray(raw_action, dtype=np.float32)
        clipped_arr = np.asarray(clipped_action, dtype=np.float32)
        self._action_abs_sum += float(np.mean(np.abs(clipped_arr)))
        self._action_clip_fraction_sum += float(
            np.mean(np.abs(raw_arr - clipped_arr) > 1e-6)
        )
        self._action_metric_count += 1


class A2CGAEAgent(A2CAgent):
    """A2C with Generalized Advantage Estimation (GAE).

    Identical to ``A2CAgent`` except for the advantage computation: GAE uses
    an exponential smoothing weighted by λ that controls the bias-variance
    tradeoff.

    λ = 0 → TD(0) (high variance, low bias)
    λ = 1 → N-step Monte Carlo (low variance, potentially higher bias)

    Parameters
    ----------
    gae_lambda:
        GAE smoothing parameter (0.95 by default — standard A2C/PPO).
    See ``A2CAgent`` for the other parameters.
    """

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
            normalize_observations=normalize_observations,
            obs_norm_epsilon=obs_norm_epsilon,
            obs_norm_clip=obs_norm_clip,
            device=device,
        )
        self.gae_lambda = gae_lambda

    def _compute_advantages(
        self, next_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override: use GAE instead of plain N-step returns.

        Parameters
        ----------
        next_value:
            V(s_{T+1}) after the end of the rollout.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(returns, advantages)`` computed by GAE, of shape ``(n_steps,)``.
        """
        return self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)


class A3CAgent(A2CGAEAgent):
    """A3C agent (Asynchronous Advantage Actor-Critic).

    Extends A2CGAEAgent by adding the shared-memory mechanics for asynchronous
    multi-process training. Each worker maintains a local model, computes
    gradients over a rollout of length t_max, then pushes those gradients to
    the shared model before a shared Adam step.

    Advantage computation is inherited from A2CGAEAgent (GAE), ensuring
    consistency with A2C-GAE.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    config:
        Optional config dataclass; the hyperparameters are extracted from it.
    hidden_dim:
        Width of each hidden layer.
    lr:
        Learning rate of SharedAdam.
    gamma:
        Discount factor.
    n_steps:
        Capacity of each worker's local buffer (alias of t_max).
    entropy_coef:
        Entropy coefficient in the total loss.
    value_coef:
        Coefficient of the value loss.
    max_grad_norm:
        Maximum norm for gradient clipping.
    gae_lambda:
        GAE λ parameter.
    num_workers:
        Number of parallel worker processes.
    t_max:
        Number of steps per rollout for each worker.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 20,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        num_workers: int = 4,
        t_max: int = 20,
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
            device=device,
        )
        self.num_workers = num_workers
        self.t_max = t_max

    def create_shared_model(self) -> tuple:
        """Create the shared-memory actor, critic, and optimizer.

        The networks are moved into shared memory (``share_memory_()``), which
        allows workers to read the weights without copying and to write the
        gradients directly into the shared tensors.

        Returns
        -------
        tuple[GaussianActor, CriticNetwork, SharedAdam]
            ``(shared_actor, shared_critic, shared_optimizer)``
        """
        shared_actor = GaussianActor(self.obs_dim, self.action_dim, self.hidden_dim)
        shared_critic = CriticNetwork(self.obs_dim, self.hidden_dim)
        shared_actor.share_memory()
        shared_critic.share_memory()
        shared_optimizer = SharedAdam(
            list(shared_actor.parameters()) + list(shared_critic.parameters()),
            lr=self.lr,
        )
        return shared_actor, shared_critic, shared_optimizer

    @staticmethod
    def sync_local_from_shared(local_model: nn.Module, shared_model: nn.Module) -> None:
        """Copy the weights from the shared model to the local model.

        Called at the start of each worker rollout to restart from the most
        recent global weights.

        Parameters
        ----------
        local_model:
            The worker's local network (actor or critic).
        shared_model:
            The network shared in common memory.
        """
        local_model.load_state_dict(shared_model.state_dict())

    @staticmethod
    def push_gradients_to_shared(
        local_model: nn.Module, shared_model: nn.Module
    ) -> None:
        """Copy the gradients from the local model to the shared model.

        Workers compute gradients on their local copies, then push them to
        the shared model so that the shared optimizer can perform its update.

        Parameters
        ----------
        local_model:
            Local network containing the computed gradients.
        shared_model:
            Shared network that will receive the gradients.
        """
        for local_p, shared_p in zip(
            local_model.parameters(), shared_model.parameters()
        ):
            shared_p._grad = local_p.grad
