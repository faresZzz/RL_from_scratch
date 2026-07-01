"""Training loop for SAC (Soft Actor-Critic).

SAC is an **off-policy maximum-entropy** actor-critic algorithm:

1. **Collect** transitions using a stochastic policy ``a ~ π_θ(·|s)`` and
   store them in a replay buffer.
2. **Update twin Q-functions** (critics) by minimising the soft Bellman
   residual:

       L_Q = 𝔼[(Q_φ(s,a) − (r + γ(Q̄(s',ã') − α log π_θ(ã'|s'))))²]

   where ``ã' ~ π_θ(·|s')``, ``Q̄`` is the minimum of two target networks
   (clipped double-Q trick), and ``α`` is the entropy temperature.

3. **Update the policy** to maximise expected return **plus** entropy:

       J_π = 𝔼_s[𝔼_{a~π}[α log π_θ(a|s) − Q(s,a)]]

4. **Adjust the temperature** α via dual gradient descent on the constraint
   ``𝔼[-log π(a|s)] ≥ H̄`` (target entropy, typically ``−dim(A)``).

The off-policy buffer allows many gradient steps per environment step.
A random warm-up phase fills the buffer before learning begins.
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
from rl_from_scratch.sac.agent import SACAgent
from rl_from_scratch.sac.config import SACConfig
import rl_from_scratch.sac.reporting as _sac_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# SAC evaluation (needs action clipping like training)
# ======================================================================

def evaluate(
    agent: SACAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate the deterministic SAC policy on a fresh environment."""
    eval_env = make_env(env_id, seed=seed, render=False)
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            total_reward = 0.0
            ep_length = 0

            for step in range(max_steps):
                # Deterministic policy: μ(s) = tanh(mean), no sampling.
                action = agent.select_action(obs, deterministic=True)
                env_action = clip_action(action, eval_env)
                obs, reward, terminated, truncated, _ = eval_env.step(env_action)
                total_reward += float(reward)
                ep_length += 1
                if terminated or truncated:
                    break

            rewards.append(total_reward)
            lengths.append(ep_length)
    finally:
        eval_env.close()

    r = np.asarray(rewards, dtype=np.float32)
    summary: dict[str, float] = {
        "mean_reward": float(r.mean()) if len(r) else 0.0,
        "std_reward": float(r.std()) if len(r) else 0.0,
        "min_reward": float(r.min()) if len(r) else 0.0,
        "max_reward": float(r.max()) if len(r) else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    if solved_reward is not None:
        summary["success_rate"] = float(
            sum(1 for x in rewards if x >= solved_reward) / max(1, n_episodes)
        )
    return summary


# ======================================================================
# Main training entry-point
# ======================================================================

@register_agent("sac")
def train_sac(
    config: SACConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a SAC agent.

    **Timestep-based loop** with off-policy replay buffer.  Updates start
    after ``config.update_after`` steps and fire every ``config.update_every``
    steps.
    """
    del render
    config = apply_overrides(config, SACConfig, seed=seed, run_name=run_name, output_dir=output_dir)

    set_all_seeds(config.seed)
    env = make_env(config.env_id, seed=config.seed, render=False)

    try:
        env.action_space.seed(config.seed)
        info = get_env_info(env)

        # SAC needs explicit action bounds for the squashed Gaussian.
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)

        agent = build_agent(
            SACAgent, config,
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
                    # Warm-up: fill replay buffer with random actions before
                    # any policy-driven exploration.
                    action = env.action_space.sample()
                else:
                    # Sample from the squashed Gaussian:
                    # a = tanh(μ_θ(s) + σ_θ(s) · ε),  ε ~ N(0,I)
                    action = agent.select_action(obs, deterministic=False)

                env_action = np.asarray(clip_action(action, env), dtype=np.float32)

                # --- Environment step ---
                next_obs, reward, terminated, truncated, _ = env.step(env_action)
                done = bool(terminated or truncated)
                terminal = bool(terminated)
                episode_reward += float(reward)
                episode_length += 1
                global_step += 1

                # Store with terminal flag (not done) so timeouts don't
                # look like failures to the Q-target.
                agent.store_transition(obs, env_action, float(reward), next_obs, terminal)

                # --- Twin-Q / actor / alpha updates ---
                # Updates begin after update_after steps, then every
                # update_every steps.  Each update:
                #   1. Sample mini-batch from replay
                #   2. Critic: minimise soft Bellman residual
                #   3. Actor: maximise Q(s,a) + α·H(π)
                #   4. Alpha: enforce target entropy ≥ H̄
                #   5. Soft-update target networks: θ̄ ← τθ + (1−τ)θ̄
                if global_step >= config.update_after and global_step % config.update_every == 0:
                    update_metrics = agent.learn_step()
                    if update_metrics:
                        # Filter to finite float values only.
                        payload = {
                            k: float(v) for k, v in update_metrics.items()
                            if isinstance(v, (int, float, np.floating, np.integer))
                            and np.isfinite(float(v))
                        }
                        if payload:
                            manager.record_updates(payload)

                obs = next_obs
                if done:
                    break

            agent.episode_ended()
            progress.update(episode_length)

            # --- Record episode ---
            manager.record_episode(reward=episode_reward, length=episode_length)

            if episode_count % 10 == 0:
                progress.set_postfix(ep=episode_count, reward=f"{episode_reward:.1f}")

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=global_step)
            manager.maybe_eval(agent, episode=episode_count, timestep=global_step)

        progress.close()
        return manager.finalize_run(agent, reporting_module=_sac_reporting, observed_timesteps=global_step)
    finally:
        env.close()
