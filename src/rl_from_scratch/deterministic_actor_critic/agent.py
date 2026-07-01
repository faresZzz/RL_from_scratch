"""DDPG and TD3 agents — off-policy deterministic actor-critic.

``DDPGAgent`` implements DDPG (Lillicrap et al., 2015) with target networks
and a replay buffer. ``TD3Agent`` inherits from ``DDPGAgent`` and overrides the
loss methods to apply the three TD3 improvements.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.diagnostics import ActionDiagnosticsMixin
from rl_from_scratch.core.utils import soft_update
from rl_from_scratch.deterministic_actor_critic.network import (
    DeterministicActor,
    ContinuousQNetwork,
    TwinQNetwork,
)
from rl_from_scratch.deterministic_actor_critic.buffer import ContinuousReplayBuffer
from rl_from_scratch.deterministic_actor_critic.noise import GaussianNoise, OUNoise


class DDPGAgent(ActionDiagnosticsMixin, BaseAgent):
    """Deep Deterministic Policy Gradient (DDPG) agent for continuous spaces.

    Maintains four networks: online actor, target actor (deepcopy), online
    critic, target critic (deepcopy). The target networks are updated softly
    after each gradient step.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the continuous action space.
    hidden_dim:
        Width of each hidden layer of the MLP networks.
    actor_lr:
        Learning rate of the actor.
    critic_lr:
        Learning rate of the critic.
    gamma:
        Discount factor for returns.
    tau:
        Soft-update coefficient for the target networks.
    buffer_capacity:
        Maximum capacity of the replay buffer.
    batch_size:
        Batch size for gradient updates.
    noise_type:
        Type of exploration noise: ``"gaussian"`` or ``"ou"``.
    noise_std:
        Standard deviation of the exploration noise.
    action_low:
        Lower bound of the action space.
    action_high:
        Upper bound of the action space.
    device:
        Compute device (``"cpu"``, ``"cuda"``, ``"auto"``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        noise_type: str = "gaussian",
        noise_std: float = 0.1,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        device: str = "auto",
    ) -> None:
        from rl_from_scratch.core.utils import resolve_device

        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        # Action bounds in numpy (for select_action/np.clip) and tensor (for TD3 clamp)
        self._action_low_np = np.asarray(action_low, dtype=np.float32)
        self._action_high_np = np.asarray(action_high, dtype=np.float32)
        self.action_low = torch.as_tensor(self._action_low_np, device=self.device)
        self.action_high = torch.as_tensor(self._action_high_np, device=self.device)

        # Online and target actor networks
        self.actor = DeterministicActor(
            obs_dim, action_dim, hidden_dim, action_low, action_high
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_target.eval()

        # Online and target critic networks (DDPG uses a single critic)
        self.critic = ContinuousQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()

        # Separate actor / critic optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Off-policy replay buffer
        self.replay_buffer = ContinuousReplayBuffer(buffer_capacity)

        # Exploration noise
        if noise_type == "ou":
            self.noise: GaussianNoise | OUNoise = OUNoise(action_dim, sigma=noise_std)
        else:
            self.noise = GaussianNoise(action_dim, sigma=noise_std)

        # Update counter (used by TD3 for the policy delay)
        self._update_step = 0
        self._last_raw_action: np.ndarray | None = None
        self._reset_action_diagnostics()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> np.ndarray:
        """Choose a continuous action via the deterministic policy μ_θ(s).

        In exploration mode, adds noise to the action and clips it to the bounds.
        In deterministic mode (evaluation), returns μ_θ(s) directly.

        Parameters
        ----------
        observation:
            Current observation from the environment.
        deterministic:
            If True, returns the action without noise (evaluation mode).

        Returns
        -------
        np.ndarray
            Action of shape ``(action_dim,)`` in [action_low, action_high].
        """
        obs_t = self._to_tensor(observation)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()

        raw_action = action
        if not deterministic:
            raw_action = action + self.noise()
            action = np.clip(raw_action, self._action_low_np, self._action_high_np)

        self._last_raw_action = np.asarray(raw_action, dtype=np.float32)
        return action.astype(np.float32)

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs:
            Observation of the current state.
        action:
            Continuous action taken (np.ndarray).
        reward:
            Reward received.
        next_obs:
            Next observation.
        done:
            True if the episode ended.
        """
        self.replay_buffer.push(obs, np.asarray(action, dtype=np.float32), reward, next_obs, done)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Perform a full DDPG update if the buffer is large enough.

        Sequence: sampling → critic loss → actor loss → soft update.

        Returns
        -------
        dict[str, float]
            Metrics: ``actor_loss``, ``critic_loss``, ``q_mean``.
            Empty if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic update
        critic_loss = self._compute_critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update and soft update of the targets
        actor_metrics = self._update_actor_and_targets(states)

        self._update_step += 1

        with torch.no_grad():
            q_mean = self.critic(states, actions).mean().item()
        diagnostics = self._consume_action_diagnostics(default_actions=actions)

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": q_mean,
            "noise_std": float(self.noise_std),
            **actor_metrics,
            **diagnostics,
        }

    def episode_ended(self) -> None:
        """No action required at the end of an episode for DDPG."""

    def save(self, path: str | Path) -> Path:
        """Save all networks and optimizers to a .pt file.

        Parameters
        ----------
        path:
            Destination path of the checkpoint.

        Returns
        -------
        Path
            Path of the created file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.hidden_dim,
                    "action_low": self._action_low_np.tolist(),
                    "action_high": self._action_high_np.tolist(),
                },
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        **kwargs: Any,
    ) -> DDPGAgent:
        """Load an agent from a checkpoint.

        The dimensions (obs_dim, action_dim, hidden_dim, action bounds) are
        read from the checkpoint metadata. Additional kwargs can override the
        agent's default values.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.
        device:
            Destination device.

        Returns
        -------
        DDPGAgent
            Agent reconstructed with the checkpoint weights.
        """
        checkpoint = torch.load(Path(path), weights_only=True, map_location="cpu")
        meta = checkpoint.get("meta", {})
        agent = cls(
            obs_dim=meta["obs_dim"],
            action_dim=meta["action_dim"],
            hidden_dim=meta.get("hidden_dim", 256),
            action_low=np.array(meta.get("action_low", [-1.0]), dtype=np.float32),
            action_high=np.array(meta.get("action_high", [1.0]), dtype=np.float32),
            device=device,
            **kwargs,
        )
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.actor_target.load_state_dict(checkpoint["actor_target"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.critic_target.load_state_dict(checkpoint["critic_target"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return agent

    # ------------------------------------------------------------------
    # Fork points for TD3
    # ------------------------------------------------------------------

    def _compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the critic loss via the DDPG Bellman target.

        Target: y = r + γ·(1-done)·Q_target(s', μ_target(s'))
        Loss: MSE(Q(s, a), y)

        Fork point: TD3Agent overrides this method to use the twin critics
        and target policy smoothing.

        Parameters
        ----------
        states, actions, rewards, next_states, dones:
            Batch of transitions from the replay buffer.

        Returns
        -------
        torch.Tensor
            Scalar critic loss.
        """
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q_current = self.critic(states, actions)
        return nn.functional.mse_loss(q_current, q_target)

    def _update_actor_and_targets(self, states: torch.Tensor) -> dict[str, Any]:
        """Update the actor and apply the soft update of the targets.

        Actor loss: -E[Q(s, μ_θ(s))]
        Soft update: θ_target ← τ·θ + (1-τ)·θ_target

        Fork point: TD3Agent overrides this method to apply the policy delay
        (actor update only every policy_delay steps).

        Parameters
        ----------
        states:
            Batch of states for computing the actor loss.

        Returns
        -------
        dict[str, Any]
            ``{"actor_loss": float, "actor_updated": True}``
        """
        # Actor loss: maximize Q(s, μ(s)) → minimize -Q(s, μ(s))
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of both target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return {"actor_loss": actor_loss.item(), "actor_updated": 1.0}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def record_action_diagnostics(self, raw_action: Any, clipped_action: Any) -> None:
        """Accumulate action magnitude and clip fraction for the next update."""
        raw_arr = np.asarray(raw_action, dtype=np.float32)
        clipped_arr = np.asarray(clipped_action, dtype=np.float32)
        self._action_abs_sum += float(np.mean(np.abs(clipped_arr)))
        self._action_clip_fraction_sum += float(
            np.mean(np.abs(raw_arr - clipped_arr) > 1e-6)
        )
        self._action_metric_count += 1


class TD3Agent(DDPGAgent):
    """Twin Delayed DDPG (TD3) agent — improved version of DDPG.

    Three improvements over DDPG (Fujimoto et al., 2018):

    1. **Twin critics**: two independent critics Q1, Q2. The Bellman target
       uses min(Q1_target, Q2_target) to reduce overestimation.

    2. **Delayed policy updates**: the actor is updated only every
       ``policy_delay`` critic steps, reducing the variance of the updates.

    3. **Target policy smoothing**: clipped Gaussian noise added to the target
       action to regularize the Bellman target (avoids exploiting peaks in Q).

    Parameters
    ----------
    policy_delay:
        Number of critic updates before each actor update.
    target_noise:
        Standard deviation of the smoothing noise added to the target action.
    target_noise_clip:
        Maximum absolute value of the clipped noise.
    See ``DDPGAgent`` for the other parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        noise_type: str = "gaussian",
        noise_std: float = 0.1,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        device: str = "auto",
    ) -> None:
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            noise_type=noise_type,
            noise_std=noise_std,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        # Replace the DDPG critic with a twin Q-network
        self.critic = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # TD3-specific parameters
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

    # ------------------------------------------------------------------
    # Overrides of the DDPG fork points
    # ------------------------------------------------------------------

    def _compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the twin critics loss with target policy smoothing.

        TD3 target:
        1. Compute the target action μ_target(s') + clipped noise.
        2. Clip the target action to the bounds of the action space.
        3. y = r + γ·(1-done)·min(Q1_target(s', a'), Q2_target(s', a'))
        4. Loss = MSE(Q1(s, a), y) + MSE(Q2(s, a), y)

        Parameters
        ----------
        states, actions, rewards, next_states, dones:
            Batch of transitions from the replay buffer.

        Returns
        -------
        torch.Tensor
            Combined scalar loss of the two critics.
        """
        with torch.no_grad():
            # Target action with clipped smoothing noise
            next_actions = self.actor_target(next_states)
            noise = (
                torch.randn_like(next_actions) * self.target_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (next_actions + noise).clamp(
                self.action_low, self.action_high
            )

            # Min of the two target Q values to reduce overestimation
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        # Loss of both critics simultaneously
        q1_current, q2_current = self.critic(states, actions)
        loss_q1 = nn.functional.mse_loss(q1_current, q_target)
        loss_q2 = nn.functional.mse_loss(q2_current, q_target)
        return loss_q1 + loss_q2

    def _update_actor_and_targets(self, states: torch.Tensor) -> dict[str, Any]:
        """Update the actor with policy delay and apply the soft updates.

        The actor is updated only every ``policy_delay`` critic steps.
        The actor loss uses only Q1 (via ``q1_forward``) to avoid computing
        Q2 unnecessarily.

        Parameters
        ----------
        states:
            Batch of states for computing the actor loss.

        Returns
        -------
        dict[str, Any]
            Metrics including ``actor_loss``, ``actor_updated``, ``q1_mean``,
            ``q2_mean``, ``q_gap``, ``target_q_mean``.
        """
        metrics: dict[str, Any] = {"actor_updated": 0.0}

        # Delayed actor update
        if (self._update_step + 1) % self.policy_delay == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update of both target networks
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_updated"] = 1.0
        else:
            metrics["actor_loss"] = float("nan")

        return metrics

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Perform a full TD3 update and return enriched metrics.

        Extends the DDPG method with the twin-critics-specific metrics:
        q1_mean, q2_mean, q_gap, target_q_mean.

        Returns
        -------
        dict[str, float]
            Metrics: ``critic_loss``, ``q_mean``, ``actor_loss``,
            ``q1_mean``, ``q2_mean``, ``q_gap``, ``target_q_mean``.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic update
        critic_loss = self._compute_critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (with delay) and soft updates
        actor_metrics = self._update_actor_and_targets(states)

        self._update_step += 1

        # Twin-critics-specific metrics
        with torch.no_grad():
            q1_vals, q2_vals = self.critic(states, actions)
            q1_mean = q1_vals.mean().item()
            q2_mean = q2_vals.mean().item()
            q_gap = abs(q1_mean - q2_mean)

            next_actions_tgt = self.actor_target(next_states)
            q1_next_tgt, q2_next_tgt = self.critic_target(next_states, next_actions_tgt)
            target_q_mean = torch.min(q1_next_tgt, q2_next_tgt).mean().item()
        diagnostics = self._consume_action_diagnostics(default_actions=actions)

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": (q1_mean + q2_mean) / 2.0,
            "q1_mean": q1_mean,
            "q2_mean": q2_mean,
            "q_gap": q_gap,
            "target_q_mean": target_q_mean,
            "noise_std": float(self.noise_std),
            **actor_metrics,
            **diagnostics,
        }

    def save(self, path: str | Path) -> Path:
        """Save all TD3 networks and optimizers to a .pt file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "meta": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden_dim": self.hidden_dim,
                    "action_low": self._action_low_np.tolist(),
                    "action_high": self._action_high_np.tolist(),
                },
            },
            output_path,
        )
        return output_path
