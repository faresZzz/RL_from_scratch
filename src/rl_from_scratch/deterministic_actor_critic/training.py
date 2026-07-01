"""Training loops for DDPG and TD3.

Both algorithms use the **deterministic policy gradient** (DPG) framework:

1. **Collect** transitions using a noisy deterministic policy
   ``a = μ_θ(s) + noise`` and store in a replay buffer.
2. **Update the critic** Q_φ by minimising the Bellman residual:

       L_Q = 𝔼[(Q_φ(s,a) − (r + γ Q_φ̄(s', μ_θ̄(s'))))²]

   where ``φ̄``, ``θ̄`` are slowly-moving target network weights.

3. **Update the actor** μ_θ by ascending the deterministic policy gradient:

       ∇_θ J ≈ 𝔼[∇_a Q_φ(s,a)|_{a=μ_θ(s)} · ∇_θ μ_θ(s)]

4. **Soft-update** target networks: ``θ̄ ← τθ + (1−τ)θ̄``.

**TD3** adds three improvements over DDPG:
- *Twin critics*: take ``min(Q₁, Q₂)`` in the target to reduce overestimation.
- *Delayed actor updates*: update μ_θ every ``d`` critic steps.
- *Target policy smoothing*: add clipped noise to the target action
  ``ã' = clip(μ_θ̄(s') + clip(ε, −c, c))`` to prevent sharp Q peaks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import clip_action, get_env_info, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import set_all_seeds
from rl_from_scratch.deterministic_actor_critic.agent import DDPGAgent, TD3Agent
from rl_from_scratch.deterministic_actor_critic.config import DDPGConfig, TD3Config
import rl_from_scratch.deterministic_actor_critic.reporting as _dac_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# Deterministic policy evaluation
# ======================================================================

def evaluate(
    agent: DDPGAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate the deterministic policy μ_θ(s) without exploration noise."""
    eval_env = make_env(env_id, seed=seed, render=False)
    rewards: list[float] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            total_reward = 0.0

            for step in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                env_action = clip_action(action, eval_env)
                obs, reward, terminated, truncated, _ = eval_env.step(env_action)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)
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
        summary["success_rate"] = float(
            sum(1 for x in rewards if x >= solved_reward) / max(1, n_episodes)
        )
    return summary


# ======================================================================
# Shared training driver (DDPG / TD3)
# ======================================================================

def _train_deterministic_ac(
    *,
    config: DDPGConfig | TD3Config,
    agent_cls: type[DDPGAgent],
    config_cls: type[DDPGConfig] | type[TD3Config],
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

        # Continuous action bounds for clipping and agent initialisation.
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)

        agent = build_agent(
            agent_cls, config,
            obs_dim=info["obs_dim"],
            action_dim=info["action_dim"],
            action_low=action_low,
            action_high=action_high,
        )
        manager = RunManager.from_config(config, agent=agent, evaluate_fn=evaluate)

        global_step = 0
        episode_count = 0

        logger.info("Starting %s on %s (%d timesteps)", config.approach, config.env_id, config.total_timesteps)
        progress = tqdm(total=config.total_timesteps, desc=config.approach, mininterval=1.0)

        while global_step < config.total_timesteps:
            episode_count += 1
            obs, _ = env.reset(seed=config.seed + episode_count - 1)
            episode_reward = 0.0
            episode_length = 0

            for step in range(min(config.max_steps_per_episode, config.total_timesteps - global_step)):
                # --- Action selection ---
                if global_step < config.start_steps:
                    # Warm-up: random actions to pre-fill the replay buffer.
                    action = env.action_space.sample()
                    raw_action = action
                else:
                    # Deterministic policy + exploration noise:
                    # a = μ_θ(s) + N(0, σ)  (OU or Gaussian noise).
                    action = agent.select_action(obs, deterministic=False)
                    # The agent stores the pre-clip action internally.
                    raw_action = agent._last_raw_action

                env_action = np.asarray(clip_action(action, env), dtype=np.float32)

                # --- Environment step ---
                next_obs, reward, terminated, truncated, _ = env.step(env_action)
                done = bool(terminated or truncated)
                terminal = bool(terminated)
                episode_reward += float(reward)
                episode_length += 1
                global_step += 1

                agent.record_action_diagnostics(raw_action=raw_action, clipped_action=env_action)
                agent.store_transition(obs, env_action, float(reward), next_obs, terminal)

                # --- Off-policy gradient updates ---
                # After update_after steps, every update_every steps:
                #   DDPG: one critic + one actor gradient step.
                #   TD3:  one twin-critic step; actor only every d steps
                #         (delayed policy updates).
                if global_step >= config.update_after and global_step % config.update_every == 0:
                    learn_result = agent.learn_step()
                    if learn_result:
                        payload: dict[str, float] = {}
                        for key, val in learn_result.items():
                            if not isinstance(val, (int, float, np.floating, np.integer)):
                                continue
                            value = float(val)
                            if not np.isfinite(value):
                                continue
                            # Track TD3's delayed actor update flag separately.
                            if key == "actor_updated":
                                manager.history.setdefault("step_actor_update_flags", []).append(value)
                            else:
                                payload[key] = value
                        if payload:
                            manager.record_updates(payload)

                obs = next_obs
                if done:
                    break

            agent.episode_ended()
            agent.noise.reset()
            progress.update(episode_length)

            # --- Record episode ---
            manager.record_episode(reward=episode_reward, length=episode_length)

            if episode_count % 10 == 0:
                progress.set_postfix(ep=episode_count, reward=f"{episode_reward:.1f}")

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=global_step)
            manager.maybe_eval(agent, episode=episode_count, timestep=global_step)

        progress.close()
        return manager.finalize_run(agent, reporting_module=_dac_reporting, observed_timesteps=global_step)
    finally:
        env.close()


# ======================================================================
# Public entry-points
# ======================================================================

@register_agent("ddpg")
def train_ddpg(
    config: DDPGConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a DDPG agent (Deep Deterministic Policy Gradient)."""
    del render
    return _train_deterministic_ac(
        config=config, agent_cls=DDPGAgent, config_cls=DDPGConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("td3")
def train_td3(
    config: TD3Config,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a TD3 agent (Twin Delayed DDPG)."""
    del render
    return _train_deterministic_ac(
        config=config, agent_cls=TD3Agent, config_cls=TD3Config,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )
