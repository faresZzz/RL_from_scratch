"""MBPO agent: probabilistic ensemble + SAC policy + mixed-buffer updates.

Architecture (Janner et al. 2019):
1. Collect real transitions into an environment replay buffer.
2. Fit a probabilistic ensemble that predicts (Δstate, reward).
3. Generate short imagined rollouts from the ensemble, stored in a model buffer.
4. Update the SAC policy on a mix of real and imagined transitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device
from rl_from_scratch.mbpo.buffer import ModelBuffer, ReplayBuffer, sample_mixed
from rl_from_scratch.mbpo.dynamics import ProbabilisticEnsemble
from rl_from_scratch.mbpo.sac import SacLearner


class MbpoAgent(BaseAgent):
    """Model-Based Policy Optimization agent.

    Parameters
    ----------
    obs_dim:
        Observation space dimensionality.
    action_dim:
        Action space dimensionality.
    action_low:
        Lower bound of the action space.
    action_high:
        Upper bound of the action space.
    seed:
        Random seed.
    device:
        Compute device (``"auto"``, ``"cpu"``, etc.).

    The remaining keyword parameters come from ``MbpoConfig`` fields and are
    wired automatically by ``build_agent``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        action_low: Any = -1.0,
        action_high: Any = 1.0,
        seed: int = 0,
        device: str = "auto",
        # Ensemble hyperparams
        ensemble_size: int = 7,
        model_hidden_dim: int = 200,
        model_n_layers: int = 4,
        model_lr: float = 1e-3,
        model_fit_steps: int = 200,
        model_batch_size: int = 256,
        weight_decay: float = 1e-4,
        # SAC hyperparams
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        sac_hidden_dim: int = 256,
        sac_batch_size: int = 256,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        target_entropy: float | None = None,
        # MBPO rollout hyperparams
        rollout_length: int = 1,
        rollout_batch_size: int = 400,
        rollout_every: int = 50,
        updates_per_step: int = 20,
        real_ratio: float = 0.05,
        env_buffer_capacity: int = 1_000_000,
        model_buffer_capacity: int = 400_000,
        num_warmup_steps: int = 1_000,
        # Additional config fields forwarded by build_agent (ignored here)
        **_kwargs: Any,
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._action_low = np.asarray(action_low, dtype=np.float32)
        self._action_high = np.asarray(action_high, dtype=np.float32)
        self.seed = seed

        # Store hyperparams for save/load
        self.model_lr = model_lr
        self.model_fit_steps = model_fit_steps
        self.model_batch_size = model_batch_size
        self.weight_decay = weight_decay
        self.sac_batch_size = sac_batch_size
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.rollout_every = rollout_every
        self.updates_per_step = updates_per_step
        self.real_ratio = real_ratio
        self.num_warmup_steps = num_warmup_steps

        # Probabilistic ensemble predicts [Δstate (obs_dim), reward (1)]
        self.ensemble = ProbabilisticEnsemble(
            input_dim=obs_dim + action_dim,
            output_dim=obs_dim + 1,
            ensemble_size=ensemble_size,
            hidden_dim=model_hidden_dim,
            n_layers=model_n_layers,
        ).to(self.device)

        # SAC learner (no internal buffer)
        self.sac = SacLearner(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=sac_hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            auto_tune_alpha=auto_tune_alpha,
            alpha_lr=3e-4,
            target_entropy=target_entropy,
            action_low=self._action_low,
            action_high=self._action_high,
            device=device,
        )

        # Replay buffers
        self.env_buffer = ReplayBuffer(env_buffer_capacity)
        self.model_buffer = ModelBuffer(model_buffer_capacity)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: Any, *, deterministic: bool = False) -> Any:
        """Choose an action.

        Returns a uniform random action while the environment buffer is
        smaller than ``num_warmup_steps`` (warm-up phase).
        """
        if len(self.env_buffer) < self.num_warmup_steps:
            return (
                np.random.uniform(self._action_low, self._action_high)
                .astype(np.float32)
            )
        return self.sac.select_action(observation, deterministic=deterministic)

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Push a real environment transition into the env buffer."""
        self.env_buffer.push(obs, action, reward, next_obs, done)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Perform ``updates_per_step`` SAC updates on mixed batches.

        Returns averaged finite metrics.  Returns an empty dict if neither
        buffer has enough data.
        """
        total = self.sac_batch_size
        if len(self.env_buffer) < total and len(self.model_buffer) < total:
            return {}
        # Need at least 1 sample from a non-empty buffer
        if len(self.env_buffer) == 0 and len(self.model_buffer) < total:
            return {}
        if len(self.model_buffer) == 0 and len(self.env_buffer) < total:
            return {}

        accumulated: dict[str, float] = {}
        valid_updates = 0

        for _ in range(self.updates_per_step):
            try:
                obs, act, rew, next_obs, done = sample_mixed(
                    self.env_buffer,
                    self.model_buffer,
                    self.sac_batch_size,
                    self.real_ratio,
                )
            except (RuntimeError, ValueError):
                continue

            obs = obs.to(self.device)
            act = act.to(self.device)
            rew = rew.to(self.device)
            next_obs = next_obs.to(self.device)
            done = done.to(self.device)

            metrics = self.sac.update(obs, act, rew, next_obs, done)

            for k, v in metrics.items():
                accumulated[k] = accumulated.get(k, 0.0) + v
            valid_updates += 1

        if valid_updates == 0:
            return {}

        return {k: v / valid_updates for k, v in accumulated.items()}

    def episode_ended(self) -> None:
        """No-op: MBPO uses a step-based loop, not episode-based."""

    def save(self, path: Path) -> Path:
        """Persist ensemble and SAC state dicts + hyperparams to a .pt file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "ensemble": self.ensemble.state_dict(),
            "sac": self.sac.state_dict(),
            "meta": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_low": self._action_low.tolist(),
                "action_high": self._action_high.tolist(),
                "seed": self.seed,
                "model_lr": self.model_lr,
                "model_fit_steps": self.model_fit_steps,
                "model_batch_size": self.model_batch_size,
                "weight_decay": self.weight_decay,
                "sac_batch_size": self.sac_batch_size,
                "rollout_length": self.rollout_length,
                "rollout_batch_size": self.rollout_batch_size,
                "rollout_every": self.rollout_every,
                "updates_per_step": self.updates_per_step,
                "real_ratio": self.real_ratio,
                "num_warmup_steps": self.num_warmup_steps,
                "ensemble_size": self.ensemble.ensemble_size,
                "model_hidden_dim": self.ensemble.hidden_dim,
                "model_n_layers": self.ensemble.n_layers,
                "sac_hidden_dim": self.sac.actor.trunk[0].out_features,
            },
        }

        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "MbpoAgent":
        """Restore an agent from a checkpoint file.

        Parameters
        ----------
        path:
            Path to a ``.pt`` checkpoint created by ``save``.
        **kwargs:
            Override meta-parameters (e.g. ``device``).
        """
        ckpt = torch.load(Path(path), weights_only=False, map_location="cpu")
        meta = ckpt.get("meta", {})
        device = kwargs.pop("device", "cpu")

        agent = cls(
            obs_dim=meta["obs_dim"],
            action_dim=meta["action_dim"],
            action_low=np.array(meta.get("action_low", [-1.0]), dtype=np.float32),
            action_high=np.array(meta.get("action_high", [1.0]), dtype=np.float32),
            seed=meta.get("seed", 0),
            device=device,
            ensemble_size=meta.get("ensemble_size", 7),
            model_hidden_dim=meta.get("model_hidden_dim", 200),
            model_n_layers=meta.get("model_n_layers", 4),
            model_lr=meta.get("model_lr", 1e-3),
            model_fit_steps=meta.get("model_fit_steps", 200),
            model_batch_size=meta.get("model_batch_size", 256),
            weight_decay=meta.get("weight_decay", 1e-4),
            sac_hidden_dim=meta.get("sac_hidden_dim", 256),
            sac_batch_size=meta.get("sac_batch_size", 256),
            real_ratio=meta.get("real_ratio", 0.05),
            num_warmup_steps=meta.get("num_warmup_steps", 1000),
            rollout_length=meta.get("rollout_length", 1),
            rollout_batch_size=meta.get("rollout_batch_size", 400),
            rollout_every=meta.get("rollout_every", 50),
            updates_per_step=meta.get("updates_per_step", 20),
            **kwargs,
        )

        agent.ensemble.load_state_dict(ckpt["ensemble"])
        agent.sac.load_state_dict(ckpt["sac"])
        return agent

    # ------------------------------------------------------------------
    # MBPO-specific helpers
    # ------------------------------------------------------------------

    def fit_model(self) -> dict[str, float]:
        """Fit the ensemble on all data in the env buffer.

        Returns
        -------
        dict with ``model_nll`` (mean training NLL) and
        ``ensemble_disagreement`` (epistemic uncertainty).
        """
        if len(self.env_buffer) < 2:
            return {"model_nll": 0.0, "ensemble_disagreement": 0.0}

        # Build X = [obs, act], Y = [Δstate, reward] from env buffer
        buf = self.env_buffer._buffer
        obs_arr = np.stack([t[0] for t in buf])         # [N, obs_dim]
        act_arr = np.stack([t[1] for t in buf])         # [N, act_dim]
        rew_arr = np.array([t[2] for t in buf])         # [N]
        nobs_arr = np.stack([t[3] for t in buf])        # [N, obs_dim]

        X = torch.tensor(
            np.concatenate([obs_arr, act_arr], axis=1), dtype=torch.float32
        ).to(self.device)
        delta = nobs_arr - obs_arr                      # [N, obs_dim]
        Y = torch.tensor(
            np.concatenate([delta, rew_arr[:, None]], axis=1), dtype=torch.float32
        ).to(self.device)

        mean_nll = self.ensemble.fit(
            X, Y,
            steps=self.model_fit_steps,
            batch_size=self.model_batch_size,
            lr=self.model_lr,
            weight_decay=self.weight_decay,
        )

        # Compute disagreement on a small sample to avoid OOM
        sample_size = min(256, len(self.env_buffer))
        indices = np.random.choice(len(buf), size=sample_size, replace=False)
        X_sample = X[indices]
        disagreement = self.ensemble.disagreement(X_sample)

        return {
            "model_nll": float(mean_nll),
            "ensemble_disagreement": float(disagreement),
        }

    def generate_model_rollouts(self, rollout_length: int) -> dict[str, float]:
        """Generate imagined transitions and push them to the model buffer.

        Samples ``rollout_batch_size`` start states from the env buffer, then
        rolls out for ``rollout_length`` steps using a random ensemble member
        per row per step.

        Returns
        -------
        dict with ``model_buffer_size`` and ``imagined_reward_mean``.
        """
        if len(self.env_buffer) == 0:
            return {"model_buffer_size": 0.0, "imagined_reward_mean": 0.0}

        buf = self.env_buffer._buffer
        buf_size = len(buf)
        batch_size = min(self.rollout_batch_size, buf_size)

        # Sample start states from env buffer
        indices = np.random.choice(buf_size, size=batch_size, replace=True)
        states_np = np.stack([buf[i][0] for i in indices])  # [B, obs_dim]
        states = torch.tensor(states_np, dtype=torch.float32).to(self.device)

        total_rewards: list[float] = []

        with torch.no_grad():
            for _ in range(rollout_length):
                # Sample actions from the current SAC policy (batched over all rows)
                actions, _, _ = self.sac.actor.sample(states)  # [B, act_dim]

                # Random ensemble member per row
                model_idx = torch.randint(
                    0, self.ensemble.ensemble_size, (states.shape[0],),
                    device=self.device
                )

                next_states, rewards = self.ensemble.propagate(states, actions, model_idx)

                # Push to model buffer
                states_np_step = states.cpu().numpy()
                actions_np = actions.cpu().numpy()
                rewards_np = rewards.cpu().numpy()
                next_states_np = next_states.cpu().numpy()

                for j in range(states.shape[0]):
                    self.model_buffer.push(
                        states_np_step[j],
                        actions_np[j],
                        float(rewards_np[j]),
                        next_states_np[j],
                        False,  # done=False for imagined transitions
                    )

                total_rewards.extend(rewards_np.tolist())
                states = next_states

        imagined_reward_mean = float(np.mean(total_rewards)) if total_rewards else 0.0
        return {
            "model_buffer_size": float(len(self.model_buffer)),
            "imagined_reward_mean": imagined_reward_mean,
        }
