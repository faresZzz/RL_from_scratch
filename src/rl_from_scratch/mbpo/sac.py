"""SAC learner driven by explicit batches (MBPO-internal component).

Key design difference from the standalone ``sac`` package: ``SacLearner``
does NOT own a replay buffer.  Its ``update`` method takes explicit batch
tensors so that the MBPO agent can mix real and imagined transitions
before passing them here.
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

from rl_from_scratch.core.utils import resolve_device, soft_update
from rl_from_scratch.mbpo.network import SquashedGaussianActor, TwinQNetwork


class SacLearner:
    """Entropy-regularised actor-critic that operates on explicit batch tensors.

    All gradient steps are driven from the outside via ``update(obs, act, rew,
    next_obs, done)`` — there is no internal replay buffer.

    Parameters
    ----------
    obs_dim:
        Observation dimensionality.
    action_dim:
        Action dimensionality.
    hidden_dim:
        Hidden layer width for both actor and critic.
    actor_lr:
        Adam learning rate for the actor.
    critic_lr:
        Adam learning rate for the twin critics.
    gamma:
        Discount factor.
    tau:
        Polyak soft-update coefficient.
    alpha:
        Initial entropy coefficient.
    auto_tune_alpha:
        If True, learn α via a Lagrangian constraint.
    alpha_lr:
        Adam learning rate for the log-alpha parameter.
    target_entropy:
        Target entropy.  ``None`` defaults to ``-action_dim``.
    log_std_min:
        Minimum log-std for the actor.
    log_std_max:
        Maximum log-std for the actor.
    action_low:
        Lower bound of the action space.
    action_high:
        Upper bound of the action space.
    device:
        Compute device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
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
        self.gamma = gamma
        self.tau = tau
        self.auto_tune_alpha = auto_tune_alpha
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self._action_low_np = np.asarray(action_low, dtype=np.float32)
        self._action_high_np = np.asarray(action_high, dtype=np.float32)

        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else float(-action_dim)
        )

        # Actor
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

        # Twin critics + target (no grad on target)
        self.critic = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        # Alpha
        if auto_tune_alpha:
            self.log_alpha = nn.Parameter(
                torch.log(torch.tensor(alpha, dtype=torch.float32))
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        else:
            self._alpha = float(alpha)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Current entropy coefficient α."""
        if self.auto_tune_alpha:
            return float(self.log_alpha.exp().item())
        return self._alpha

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: Any, *, deterministic: bool = False) -> np.ndarray:
        """Select an action for a single observation.

        Parameters
        ----------
        obs:
            Current observation (np.ndarray or array-like).
        deterministic:
            If True, return tanh(μ) rescaled (greedy evaluation mode).

        Returns
        -------
        np.ndarray
            Action of shape (action_dim,) clipped to action bounds.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(obs_t)
                action = (
                    torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
                )
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # SAC gradient update (operates on explicit batch tensors)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, float]:
        """One SAC gradient step on the given batch.

        Parameters
        ----------
        obs, act, rew, next_obs, done:
            Batch tensors already on the correct device.  ``rew`` and
            ``done`` should be shape (B,); the rest (B, dim).

        Returns
        -------
        dict[str, float]
            Finite metrics: critic_loss, actor_loss, alpha_loss, alpha,
            entropy, q_mean.
        """
        alpha_val = self.alpha

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            q1_targ, q2_targ = self.critic_target(next_obs, next_actions)
            min_q_targ = torch.min(q1_targ, q2_targ)
            target_q = rew + self.gamma * (1.0 - done) * (
                min_q_targ - alpha_val * next_log_probs
            )

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update ─────────────────────────────────────────────────
        actions_pi, log_probs, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, actions_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha_val * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha update ─────────────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.auto_tune_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = float(alpha_loss.item())

        # ── Soft update of critic target ─────────────────────────────────
        soft_update(self.critic_target, self.critic, self.tau)

        q_mean = float(((q1 + q2) / 2.0).mean().item())

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha,
            "entropy": float(-log_probs.mean().item()),
            "q_mean": q_mean,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable state dict."""
        d: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "meta": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
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
            d["log_alpha"] = self.log_alpha.data.clone()
            d["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return d

    def load_state_dict(self, checkpoint: dict[str, Any]) -> None:
        """Restore state from a previously saved dict."""
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.auto_tune_alpha and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
