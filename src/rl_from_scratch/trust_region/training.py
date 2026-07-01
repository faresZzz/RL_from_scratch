"""Training loops for trust-region policy optimisation (TRPO, PPO).

Both algorithms follow the **on-policy actor-critic** pattern:

1. **Collect** a rollout of transitions using the current policy π_θ.
2. **Compute advantages** Â_t = Σ (γλ)^k δ_{t+k}  (GAE) where
   δ_t = r_t + γ V(s_{t+1}) − V(s_t) is the TD residual.
3. **Update the policy** under a trust-region constraint:

   - **TRPO**: solve ``max_θ  𝔼[π_θ(a|s)/π_old(a|s) · Â]``
     subject to ``KL(π_old ‖ π_θ) ≤ δ`` via conjugate gradient + line
     search.
   - **PPO**: clip the ratio ``r = π_θ / π_old`` to ``[1−ε, 1+ε]`` and
     maximise ``min(r·Â, clip(r, 1−ε, 1+ε)·Â)``.

4. **Update the value function** V_φ to minimise ``(V_φ(s) − G_t)²``.

The rollout buffer, advantage estimation, and trust-region mechanics are
handled inside the agent's ``learn_step()`` — the training loop only
orchestrates collect → learn.  Truncated episodes bootstrap from
``V(s_{T+1})`` to avoid treating time-limits as terminal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import clip_action, get_env_info, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import set_all_seeds
from rl_from_scratch.trust_region.agent import PPOAgent, TRPOAgent, TrustRegionAgent
from rl_from_scratch.trust_region.config import PPOConfig, TRPOConfig, TrustRegionConfig
import rl_from_scratch.trust_region.reporting as _trust_region_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# Observation normalisation helper
# ======================================================================

def _normalize_observation(
    agent: TrustRegionAgent,
    observation: Any,
    *,
    update: bool,
) -> np.ndarray:
    """Apply running-mean normalisation if the agent has an obs_normalizer."""
    obs = np.asarray(observation, dtype=np.float32)
    if agent.obs_normalizer is not None:
        return np.asarray(agent.obs_normalizer.normalize(obs, update=update), dtype=np.float32)
    return obs


# ======================================================================
# One on-policy episode
# ======================================================================

@dataclass
class EpisodeResult:
    reward: float
    length: int
    seed: int | None
    terminated: bool
    truncated: bool
    learn_results: list[dict[str, float]]


def train_one_episode(
    agent: TrustRegionAgent,
    env: Any,
    *,
    max_steps: int,
    seed: int | None = None,
    deterministic: bool = False,
) -> EpisodeResult:
    """Collect one on-policy episode, triggering updates when the buffer is full.

    The inner loop follows the on-policy actor-critic pattern:
      1. Sample action from π_θ(a|s) (with log-prob and value cached).
      2. Step the environment.
      3. Store the transition in the rollout buffer.
      4. When the buffer is full → compute advantages (GAE) and run
         the trust-region update (TRPO or PPO).

    Truncated episodes (TimeLimit) bootstrap V(s') instead of treating them
    as failures: ``r_adjusted = r + γ V(s')`` when truncated but not
    terminated.
    """
    train_mode = not deterministic
    observation, _ = env.reset(seed=seed)
    observation = _normalize_observation(agent, observation, update=train_mode)
    total_reward = 0.0
    terminated = False
    truncated = False
    learn_results: list[dict[str, float]] = []

    for step in range(max_steps):
        # --- Sample action from π_θ(a|s) ---
        policy_action = np.asarray(
            agent.select_action(observation, deterministic=deterministic),
            dtype=np.float32,
        )
        env_action = np.asarray(clip_action(policy_action, env), dtype=np.float32)

        # --- Environment step ---
        next_observation, reward, terminated, truncated, _ = env.step(env_action)
        next_observation = _normalize_observation(agent, next_observation, update=train_mode)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        if train_mode:
            # Bootstrap value for truncated (TimeLimit) episodes:
            # r_adjusted = r + γ · V(s') avoids treating time-limits as failure.
            terminal_value = 0.0
            if truncated and not terminated:
                with torch.no_grad():
                    terminal_value = agent.critic(agent._to_tensor(next_observation)).item()

            bootstrapped_reward = float(reward)
            if truncated and not terminated:
                bootstrapped_reward += agent.gamma * terminal_value

            agent.record_action_diagnostics(raw_action=policy_action, clipped_action=env_action)

            # Store transition for the on-policy rollout buffer.
            agent.store_transition(observation, policy_action, bootstrapped_reward, next_observation, done)

            # --- Update when the rollout buffer is full ---
            # GAE advantage computation + policy/value gradient step.
            if agent.buffer.is_full():
                next_value = 0.0
                if not done:
                    with torch.no_grad():
                        next_value = agent.critic(agent._to_tensor(next_observation)).item()
                learn_results.append(agent.learn_step(next_value=next_value))

        observation = next_observation
        if done:
            break

    return EpisodeResult(
        reward=float(total_reward),
        length=step + 1,
        seed=seed,
        terminated=bool(terminated),
        truncated=bool(truncated),
        learn_results=learn_results,
    )


# ======================================================================
# Shared training driver (TRPO / PPO)
# ======================================================================

def _train_trust_region(
    *,
    config: TrustRegionConfig,
    agent_cls: type[TrustRegionAgent],
    config_cls: type[TrustRegionConfig],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
) -> dict[str, Any]:
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    set_all_seeds(config.seed)
    env = make_env(config.env_id, seed=config.seed, render=False)

    try:
        env.action_space.seed(config.seed)
        info = get_env_info(env)
        agent = build_agent(agent_cls, config, obs_dim=info["obs_dim"], action_dim=info["action_dim"])

        # Trust-region evaluate reuses train_one_episode(deterministic=True)
        # to respect obs normalisation and action clipping.
        def tr_evaluate(agent, env_id, *, n_episodes, seed, max_steps, solved_reward=None):
            rewards: list[float] = []
            eval_env = make_env(env_id, seed=seed, render=False)
            try:
                for i in range(n_episodes):
                    result = train_one_episode(agent, eval_env, max_steps=max_steps, seed=seed + i, deterministic=True)
                    rewards.append(result.reward)
            finally:
                eval_env.close()
            r = np.asarray(rewards, dtype=np.float32)
            summary: dict[str, float] = {
                "mean_reward": float(r.mean()) if len(r) else 0.0,
                "std_reward": float(r.std()) if len(r) else 0.0,
                "min_reward": float(r.min()) if len(r) else 0.0,
                "max_reward": float(r.max()) if len(r) else 0.0,
            }
            if solved_reward is not None:
                summary["success_rate"] = float(sum(1 for x in rewards if x >= solved_reward) / max(1, n_episodes))
            return summary

        manager = RunManager.from_config(config, agent=agent, evaluate_fn=tr_evaluate)

        global_step = 0
        episode_count = 0

        logger.info("Starting %s on %s (%d timesteps)", config.approach, config.env_id, config.total_timesteps)
        progress = tqdm(total=config.total_timesteps, desc=config.approach, mininterval=1.0)

        while global_step < config.total_timesteps:
            episode_count += 1

            # --- Collect one on-policy episode ---
            result = train_one_episode(
                agent, env,
                max_steps=min(config.max_steps_per_episode, config.total_timesteps - global_step),
                seed=config.seed + episode_count - 1,
            )
            global_step += result.length
            progress.update(result.length)

            # --- Record metrics ---
            manager.record_episode(reward=result.reward, length=result.length)
            for update_result in result.learn_results:
                # Map total_loss → loss for metric consistency.
                payload = dict(update_result)
                if "total_loss" in payload and "loss" not in payload:
                    payload["loss"] = payload["total_loss"]
                manager.record_updates(payload)

            if episode_count % 10 == 0:
                progress.set_postfix(ep=episode_count, reward=f"{result.reward:.1f}")

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=global_step)
            manager.maybe_eval(agent, episode=episode_count, timestep=global_step)

        progress.close()
        return manager.finalize_run(agent, reporting_module=_trust_region_reporting, observed_timesteps=global_step)
    finally:
        env.close()


# ======================================================================
# Public entry-points
# ======================================================================

@register_agent("trpo")
def train_trpo(
    config: TRPOConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a TRPO agent."""
    del render
    return _train_trust_region(
        config=config, agent_cls=TRPOAgent, config_cls=TRPOConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("ppo")
def train_ppo(
    config: PPOConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a PPO agent."""
    del render
    return _train_trust_region(
        config=config, agent_cls=PPOAgent, config_cls=PPOConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )
