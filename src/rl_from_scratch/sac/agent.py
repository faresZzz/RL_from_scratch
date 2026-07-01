"""SAC (Soft Actor-Critic) agent for continuous action spaces.

Implements SAC (Haarnoja et al., 2018) with:
- A squashed Gaussian stochastic policy (SquashedGaussianActor).
- Twin critics Q_φ(s, a) and their soft-updated target.
- A maximum entropy objective with optional automatic tuning of α.

Key structural difference from DDPG/TD3: no actor_target.
The online stochastic policy is used directly for the targets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device, soft_update
from rl_from_scratch.sac.buffer import ContinuousReplayBuffer
from rl_from_scratch.sac.network import SquashedGaussianActor, TwinQNetwork


class SACAgent(BaseAgent):
    """Soft Actor-Critic (SAC) agent for bounded continuous action spaces.

    Maintains three networks: online stochastic actor, online twin critic,
    target twin critic (deepcopy). Unlike DDPG/TD3, there is no target actor
    — the current stochastic policy is used directly in the Bellman target.

    The entropy coefficient α can be tuned automatically via a target entropy
    constraint H_target = -action_dim (heuristic recommended by the authors).

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
        Learning rate of the twin critics.
    gamma:
        Discount factor for returns.
    tau:
        Soft update coefficient of the target networks.
    buffer_capacity:
        Maximum capacity of the replay buffer.
    batch_size:
        Batch size for gradient updates.
    alpha:
        Initial entropy coefficient.
    auto_tune_alpha:
        If True, tunes α automatically via the target entropy constraint.
    alpha_lr:
        Learning rate for the automatic tuning of α.
    target_entropy:
        Target entropy. None → -action_dim (standard SAC heuristic).
    log_std_min:
        Minimum value of the actor's log standard deviation.
    log_std_max:
        Maximum value of the actor's log standard deviation.
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
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        alpha_lr: float = 3e-4,
        target_entropy: float | None = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        device: str = "auto",
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_tune_alpha = auto_tune_alpha
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Target entropy: -action_dim heuristic if not specified
        self.target_entropy = (
            target_entropy if target_entropy is not None else float(-action_dim)
        )

        # Action bounds as numpy
        self._action_low_np = np.asarray(action_low, dtype=np.float32)
        self._action_high_np = np.asarray(action_high, dtype=np.float32)

        # Online stochastic actor (no target actor in SAC)
        self.actor = SquashedGaussianActor(
            obs_dim,
            action_dim,
            hidden_dim,
            action_low=self._action_low_np,
            action_high=self._action_high_np,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        # Online and target twin critics
        self.critic = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Entropy coefficient α (fixed or auto-tuned)
        if auto_tune_alpha:
            self.log_alpha = nn.Parameter(
                torch.log(torch.tensor(alpha, dtype=torch.float32))
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        else:
            self._alpha = float(alpha)

        # Off-policy replay buffer
        self.replay_buffer = ContinuousReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # alpha property
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Return the current value of α (auto-tuned or fixed)."""
        if self.auto_tune_alpha:
            return self.log_alpha.exp().item()
        return self._alpha

    # ------------------------------------------------------------------
    # Interface BaseAgent
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> np.ndarray:
        """Choose an action via the stochastic or deterministic policy.

        In exploration mode, samples from π_θ(a|s) via the reparameterization
        trick. In deterministic mode (evaluation), returns rescaled tanh(μ_θ(s)).

        Parameters
        ----------
        observation:
            Current observation from the environment.
        deterministic:
            If True, returns the deterministic action without noise (evaluation mode).

        Returns
        -------
        np.ndarray
            Action of shape ``(action_dim,)`` in [action_low, action_high].
        """
        obs_t = self._to_tensor(observation)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(obs_t)
                action = (
                    torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
                )
                return action.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                action, _, _ = self.actor.sample(obs_t)
                return action.squeeze(0).cpu().numpy().astype(np.float32)

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
            True if the episode has terminated.
        """
        self.replay_buffer.push(
            obs, np.asarray(action, dtype=np.float32), reward, next_obs, done
        )

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Perform a full SAC update if the buffer is large enough.

        Sequence:
        1. Sampling from the replay buffer.
        2. Update of the twin critics (Bellman target with entropy).
        3. Update of the actor (maximizes Q - α·log π).
        4. Automatic update of α if enabled.
        5. Soft update of critic_target.

        Returns
        -------
        dict[str, float]
            Training metrics. Empty if the buffer is too small.
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

        alpha_val = self.alpha

        # ------------------------------------------------------------------
        # 1. Update of the twin critics
        # ------------------------------------------------------------------
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_targ, q2_targ = self.critic_target(next_states, next_actions)
            min_q_targ = torch.min(q1_targ, q2_targ)
            # The twin-Q target also penalizes future actions with too little entropy via alpha * log pi.
            target_q = rewards + self.gamma * (1.0 - dones) * (
                min_q_targ - alpha_val * next_log_probs
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------
        # 2. Update of the actor
        # ------------------------------------------------------------------
        actions_pi, log_probs, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        # The entropy objective pushes the policy toward actions that are both useful and still exploratory.
        actor_loss = (alpha_val * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------------------
        # 3. Automatic update of α (if enabled)
        # ------------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            # Tuning alpha raises or lowers the entropy pressure to track target_entropy.
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        # ------------------------------------------------------------------
        # 4. Soft update of critic_target
        # ------------------------------------------------------------------
        # The target critics track each update via tau to stabilize the off-policy bootstrap.
        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha,
            "entropy": -log_probs.mean().item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "q_gap": (q1 - q2).abs().mean().item(),
            "target_q_mean": target_q.mean().item(),
            "log_prob_mean": log_probs.mean().item(),
        }

    def episode_ended(self) -> None:
        """No action required at the end of an episode for SAC."""

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

        checkpoint: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "meta": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "auto_tune_alpha": self.auto_tune_alpha,
                "target_entropy": self.target_entropy,
                "log_std_min": self.log_std_min,
                "log_std_max": self.log_std_max,
                "action_low": self._action_low_np.tolist(),
                "action_high": self._action_high_np.tolist(),
            },
        }
        if self.auto_tune_alpha:
            checkpoint["log_alpha"] = self.log_alpha.data
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        else:
            checkpoint["meta"]["alpha"] = self._alpha

        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        **kwargs: Any,
    ) -> SACAgent:
        """Load an agent from a checkpoint.

        The dimensions and hyperparameters are read from the checkpoint
        metadata. Additional kwargs can override the agent's default
        values.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.
        device:
            Destination device.

        Returns
        -------
        SACAgent
            Agent reconstructed with the checkpoint weights.
        """
        checkpoint = torch.load(Path(path), weights_only=True, map_location="cpu")
        meta = checkpoint.get("meta", {})

        agent = cls(
            obs_dim=meta["obs_dim"],
            action_dim=meta["action_dim"],
            hidden_dim=meta.get("hidden_dim", 256),
            gamma=meta.get("gamma", 0.99),
            tau=meta.get("tau", 0.005),
            alpha=meta.get("alpha", 0.2),
            auto_tune_alpha=meta.get("auto_tune_alpha", True),
            target_entropy=meta.get("target_entropy"),
            log_std_min=meta.get("log_std_min", -20.0),
            log_std_max=meta.get("log_std_max", 2.0),
            action_low=np.array(meta.get("action_low", [-1.0]), dtype=np.float32),
            action_high=np.array(meta.get("action_high", [1.0]), dtype=np.float32),
            device=device,
            **kwargs,
        )

        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.critic_target.load_state_dict(checkpoint["critic_target"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if agent.auto_tune_alpha and "log_alpha" in checkpoint:
            agent.log_alpha.data.copy_(checkpoint["log_alpha"])
            agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        elif not agent.auto_tune_alpha:
            agent._alpha = float(meta.get("alpha", 0.2))

        return agent
