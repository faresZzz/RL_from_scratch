"""DreamerV1 agent: RSSM world-model + actor-critic in imagination.

Architecture (Hafner et al. 2020):
1. Encode observations with a flat MLP encoder.
2. Update the RSSM with real transitions → posterior latent states.
3. Reconstruct observations and predict rewards from posteriors
   (world-model loss: recon + reward MSE + KL).
4. Unroll H steps in imagination from posterior start states using the
   learned dynamics + actor.
5. Compute λ-returns; maximise them w.r.t. actor; regress critic to them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device
from rl_from_scratch.dreamer.actor_critic import Actor, Critic
from rl_from_scratch.dreamer.buffer import SequenceBuffer
from rl_from_scratch.dreamer.networks import Decoder, Encoder, RewardModel
from rl_from_scratch.dreamer.rssm import RSSM


# ──────────────────────────────────────────────────────────────────────────────
# λ-return helper
# ──────────────────────────────────────────────────────────────────────────────


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute λ-returns for an imagined trajectory.

    Parameters
    ----------
    rewards:
        ``[T, M]`` — imagined rewards at steps 0..T-1.
    values:
        ``[T, M]`` — value estimates at steps 0..T-1.
        ``values[-1]`` is the bootstrap at the end of the horizon.
    gamma:
        Discount factor.
    lam:
        λ mixing parameter (0 = TD(0), 1 = Monte-Carlo).

    Returns
    -------
    torch.Tensor
        ``[T-1, M]`` λ-returns for steps 0..T-2.
    """
    T = rewards.shape[0]
    rets: list[torch.Tensor] = [torch.empty_like(rewards[0])] * (T - 1)
    last = values[-1]
    for t in reversed(range(T - 1)):
        bootstrap = (1.0 - lam) * values[t + 1] + lam * last
        last = rewards[t + 1] + gamma * bootstrap
        rets[t] = last
    return torch.stack(rets, dim=0)  # [T-1, M]


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────


class DreamerAgent(BaseAgent):
    """Model-Based RL agent using a Recurrent State-Space Model (DreamerV1).

    Parameters match ``DreamerConfig`` field names exactly so that
    ``build_agent`` can wire them automatically.
    """

    def __init__(  # noqa: PLR0913
        self,
        obs_dim: int,
        action_dim: int,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        # RSSM / world-model
        deter_dim: int = 200,
        stoch_dim: int = 30,
        rssm_hidden_dim: int = 200,
        encoder_hidden_dim: int = 200,
        decoder_hidden_dim: int = 200,
        reward_hidden_dim: int = 200,
        embed_dim: int = 200,
        # Behaviour
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        # Learning rates
        model_lr: float = 6e-4,
        actor_lr: float = 8e-5,
        critic_lr: float = 8e-5,
        # RL
        gamma: float = 0.99,
        lambda_: float = 0.95,
        imagination_horizon: int = 15,
        free_nats: float = 3.0,
        kl_scale: float = 1.0,
        actor_entropy: float = 1e-4,
        grad_clip: float = 100.0,
        min_std: float = 0.1,
        # Training
        batch_size: int = 32,
        batch_length: int = 32,
        num_warmup_steps: int = 1_000,
        buffer_capacity: int = 200_000,
        # Device / misc
        device: str = "auto",
        **_kwargs: Any,
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)

        # Hyper-params (saved for checkpointing)
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.rssm_hidden_dim = rssm_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.reward_hidden_dim = reward_hidden_dim
        self.embed_dim = embed_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.model_lr = model_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lambda_ = lambda_
        self.imagination_horizon = imagination_horizon
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.actor_entropy = actor_entropy
        self.grad_clip = grad_clip
        self.min_std = min_std
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.num_warmup_steps = num_warmup_steps
        self.buffer_capacity = buffer_capacity

        self.feat_dim = deter_dim + stoch_dim

        # ── Networks ──────────────────────────────────────────────────────
        self.encoder = Encoder(obs_dim, encoder_hidden_dim, embed_dim).to(self.device)
        self.rssm = RSSM(
            action_dim, embed_dim, deter_dim, stoch_dim, rssm_hidden_dim, min_std
        ).to(self.device)
        self.decoder = Decoder(self.feat_dim, decoder_hidden_dim, obs_dim).to(self.device)
        self.reward_model = RewardModel(self.feat_dim, reward_hidden_dim).to(self.device)

        self.actor = Actor(
            self.feat_dim,
            action_dim,
            actor_hidden_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.critic = Critic(self.feat_dim, critic_hidden_dim).to(self.device)

        # ── Optimisers ────────────────────────────────────────────────────
        model_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_model.parameters())
        )
        self.model_optim = torch.optim.Adam(model_params, lr=model_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Buffer ────────────────────────────────────────────────────────
        self.buffer = SequenceBuffer(buffer_capacity)

        # ── Recurrent acting state ────────────────────────────────────────
        self._h: torch.Tensor | None = None
        self._z: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> np.ndarray:
        """Choose an action given the current observation.

        During warm-up (total transitions < num_warmup_steps) returns a
        uniform random action in [low, high] without advancing recurrent state.
        After warm-up, encodes the observation and advances the RSSM.
        """
        if len(self.buffer) < self.num_warmup_steps:
            # Uniform random in [low, high]; do NOT advance recurrent state
            return np.random.uniform(self._action_low, self._action_high).astype(
                np.float32
            )

        obs_t = torch.tensor(
            np.asarray(observation, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # [1, obs_dim]

        with torch.no_grad():
            # Initialise recurrent state on first post-warmup step or after episode reset
            if self._h is None:
                state = self.rssm.initial_state(1, device=self.device)
                self._h = state["deter"]
                self._z = state["stoch"]
                self._prev_action = torch.zeros(
                    1, self.action_dim, device=self.device
                )

            prev_state = {
                "deter": self._h,
                "stoch": self._z,
                "mean": self._z,
                "std": torch.ones_like(self._z),
            }
            embed = self.encoder(obs_t)  # [1, embed_dim]
            post, _ = self.rssm.obs_step(prev_state, self._prev_action, embed)

            # When deterministic, use posterior mean as the stoch state
            # so that the acting trajectory is fully reproducible.
            z_for_feat = post["mean"] if deterministic else post["stoch"]
            feat = torch.cat([post["deter"], z_for_feat], dim=-1)  # [1, feat_dim]

            if deterministic:
                action = self.actor.deterministic_action(feat)
            else:
                action, _ = self.actor.sample(feat)

            # Update recurrent state (use mean for deterministic to keep reproducibility)
            self._h = post["deter"]
            self._z = z_for_feat
            self._prev_action = action  # in [low, high]

        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Push a real transition into the sequence buffer."""
        self.buffer.add(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            bool(done),
        )
        self._step_count += 1

    def episode_ended(self) -> None:
        """Seal the current episode and reset recurrent acting state."""
        self.buffer.flush_current()
        self._reset_recurrent()

    def _reset_recurrent(self) -> None:
        """Reset the recurrent acting state WITHOUT touching the buffer.

        Used by deterministic evaluation, which must start from a fresh
        belief each episode but must not flush the training buffer.
        """
        self._h = None
        self._z = None
        self._prev_action = None

    def learn_step(self, **_: Any) -> dict[str, float]:
        """One world-model update + one behaviour update.

        Returns a dict of finite Python floats.  Returns all-zeros if the
        buffer does not yet have enough data for a batch.
        """
        zero: dict[str, float] = {
            "recon_loss": 0.0,
            "reward_loss": 0.0,
            "kl": 0.0,
            "model_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "imagined_return": 0.0,
        }

        # Guard: need enough transitions
        if len(self.buffer) < self.batch_size * self.batch_length:
            return zero

        # Try to sample; might raise if no episode is long enough
        try:
            batch = self.buffer.sample(self.batch_size, self.batch_length)
        except RuntimeError:
            return zero

        # Move to device
        obs = batch["obs"].to(self.device)        # [B, L, O]
        action = batch["action"].to(self.device)   # [B, L, A]
        reward = batch["reward"].to(self.device)   # [B, L]

        B, L, O = obs.shape
        A = action.shape[-1]

        # ── World-model update ────────────────────────────────────────────

        embed = self.encoder(obs)  # [B, L, E]

        # Previous actions (a_{-1} = 0 for t=0)
        prev_actions = torch.cat(
            [torch.zeros(B, 1, A, device=self.device), action[:, :-1]], dim=1
        )  # [B, L, A]

        state = self.rssm.initial_state(B, device=self.device)
        post_means_list: list[torch.Tensor] = []
        post_stds_list: list[torch.Tensor] = []
        prior_means_list: list[torch.Tensor] = []
        prior_stds_list: list[torch.Tensor] = []
        feats_list: list[torch.Tensor] = []
        post_deters: list[torch.Tensor] = []
        post_stochs: list[torch.Tensor] = []

        for t in range(L):
            post, prior = self.rssm.obs_step(state, prev_actions[:, t], embed[:, t])
            feats_list.append(self.rssm.get_feat(post))
            state = post
            post_means_list.append(post["mean"])
            post_stds_list.append(post["std"])
            prior_means_list.append(prior["mean"])
            prior_stds_list.append(prior["std"])
            post_deters.append(post["deter"])
            post_stochs.append(post["stoch"])

        feats_t = torch.stack(feats_list, dim=1)       # [B, L, F]
        pm = torch.stack(post_means_list, dim=1)        # [B, L, stoch_dim]
        ps = torch.stack(post_stds_list, dim=1)
        qm = torch.stack(prior_means_list, dim=1)
        qs = torch.stack(prior_stds_list, dim=1)

        # Reconstruction loss (MSE)
        obs_pred = self.decoder(feats_t)               # [B, L, O]
        recon_loss = 0.5 * ((obs_pred - obs) ** 2).sum(dim=-1).mean()

        # Reward prediction loss
        rew_pred = self.reward_model(feats_t).squeeze(-1)  # [B, L]
        reward_loss = 0.5 * ((rew_pred - reward) ** 2).mean()

        # KL loss (free-nats clamped)
        q_dist = torch.distributions.Normal(pm, ps)
        p_dist = torch.distributions.Normal(qm, qs)
        kl_raw = torch.distributions.kl_divergence(q_dist, p_dist).sum(dim=-1)  # [B, L]
        kl = torch.clamp(kl_raw, min=self.free_nats).mean()

        model_loss = recon_loss + reward_loss + self.kl_scale * kl

        self.model_optim.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_model.parameters()),
            self.grad_clip,
        )
        self.model_optim.step()

        # ── Behaviour update (imagination) ────────────────────────────────
        # Start states: flatten posteriors B*L, DETACH from world-model graph
        flat_deter = torch.stack(post_deters, dim=1).reshape(B * L, -1).detach()
        flat_stoch = torch.stack(post_stochs, dim=1).reshape(B * L, -1).detach()

        img_state: dict[str, torch.Tensor] = {
            "deter": flat_deter,
            "stoch": flat_stoch,
            "mean": flat_stoch.clone(),
            "std": torch.ones_like(flat_stoch),
        }

        # Imagine H steps; collect features, log-probs
        feats_img: list[torch.Tensor] = [self.rssm.get_feat(img_state)]
        log_probs_img: list[torch.Tensor] = []

        for _ in range(self.imagination_horizon):
            act_img, logp = self.actor.sample(self.rssm.get_feat(img_state))
            log_probs_img.append(logp)                          # [M]
            img_state = self.rssm.img_step(img_state, act_img)
            feats_img.append(self.rssm.get_feat(img_state))

        feats_img_t = torch.stack(feats_img, dim=0)             # [H+1, M, F]
        rewards_img = self.reward_model(feats_img_t).squeeze(-1)  # [H+1, M]
        values_img = self.critic(feats_img_t).squeeze(-1)        # [H+1, M]

        rets = lambda_returns(
            rewards_img, values_img, self.gamma, self.lambda_
        )  # [H, M]

        # Actor: maximise λ-returns + small entropy regularisation
        actor_loss = -rets.mean()
        if log_probs_img and self.actor_entropy > 0.0:
            entropy = -torch.stack(log_probs_img, dim=0).mean()
            actor_loss = actor_loss - self.actor_entropy * entropy

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        # Critic: regress to detached returns
        values_pred = self.critic(feats_img_t[:-1].detach()).squeeze(-1)  # [H, M]
        critic_loss = 0.5 * ((values_pred - rets.detach()) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        def _safe(t: torch.Tensor) -> float:
            v = float(t.detach().cpu().item())
            return v if math.isfinite(v) else 0.0

        return {
            "recon_loss": _safe(recon_loss),
            "reward_loss": _safe(reward_loss),
            "kl": _safe(kl),
            "model_loss": _safe(model_loss),
            "actor_loss": _safe(actor_loss),
            "critic_loss": _safe(critic_loss),
            "imagined_return": _safe(rets.mean()),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Persist all networks + optimisers + hyperparams to a .pt file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "model_optim": self.model_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "meta": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self._action_low.tolist(),
                "action_high": self._action_high.tolist(),
                "deter_dim": self.deter_dim,
                "stoch_dim": self.stoch_dim,
                "rssm_hidden_dim": self.rssm_hidden_dim,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "reward_hidden_dim": self.reward_hidden_dim,
                "embed_dim": self.embed_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
                "model_lr": self.model_lr,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "gamma": self.gamma,
                "lambda_": self.lambda_,
                "imagination_horizon": self.imagination_horizon,
                "free_nats": self.free_nats,
                "kl_scale": self.kl_scale,
                "actor_entropy": self.actor_entropy,
                "grad_clip": self.grad_clip,
                "min_std": self.min_std,
                "batch_size": self.batch_size,
                "batch_length": self.batch_length,
                "num_warmup_steps": self.num_warmup_steps,
                "buffer_capacity": self.buffer_capacity,
            },
        }
        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "DreamerAgent":
        """Restore an agent from a checkpoint created by ``save``."""
        ckpt = torch.load(Path(path), weights_only=False, map_location="cpu")
        meta = ckpt.get("meta", {})
        device = kwargs.pop("device", "cpu")

        agent = cls(
            obs_dim=meta["obs_dim"],
            action_dim=meta["action_dim"],
            action_low=np.array(meta.get("action_low", [-1.0]), dtype=np.float32),
            action_high=np.array(meta.get("action_high", [1.0]), dtype=np.float32),
            deter_dim=meta.get("deter_dim", 200),
            stoch_dim=meta.get("stoch_dim", 30),
            rssm_hidden_dim=meta.get("rssm_hidden_dim", 200),
            encoder_hidden_dim=meta.get("encoder_hidden_dim", 200),
            decoder_hidden_dim=meta.get("decoder_hidden_dim", 200),
            reward_hidden_dim=meta.get("reward_hidden_dim", 200),
            embed_dim=meta.get("embed_dim", 200),
            actor_hidden_dim=meta.get("actor_hidden_dim", 256),
            critic_hidden_dim=meta.get("critic_hidden_dim", 256),
            model_lr=meta.get("model_lr", 6e-4),
            actor_lr=meta.get("actor_lr", 8e-5),
            critic_lr=meta.get("critic_lr", 8e-5),
            gamma=meta.get("gamma", 0.99),
            lambda_=meta.get("lambda_", 0.95),
            imagination_horizon=meta.get("imagination_horizon", 15),
            free_nats=meta.get("free_nats", 3.0),
            kl_scale=meta.get("kl_scale", 1.0),
            actor_entropy=meta.get("actor_entropy", 1e-4),
            grad_clip=meta.get("grad_clip", 100.0),
            min_std=meta.get("min_std", 0.1),
            batch_size=meta.get("batch_size", 32),
            batch_length=meta.get("batch_length", 32),
            num_warmup_steps=meta.get("num_warmup_steps", 1000),
            buffer_capacity=meta.get("buffer_capacity", 200_000),
            device=device,
            **kwargs,
        )

        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.rssm.load_state_dict(ckpt["rssm"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.reward_model.load_state_dict(ckpt["reward_model"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.model_optim.load_state_dict(ckpt["model_optim"])
        agent.actor_optim.load_state_dict(ckpt["actor_optim"])
        agent.critic_optim.load_state_dict(ckpt["critic_optim"])
        return agent
