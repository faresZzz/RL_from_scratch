"""Training loop for DreamerV1 (Hafner et al. 2020 — Dream to Control).

Main loop:
1. Warm-up: collect ``num_warmup_steps`` transitions with random actions.
2. Per-epoch × per-step:
   a. Act in the real env with the Dreamer policy.
   b. Store the transition in the sequence buffer.
   c. Every ``train_every`` steps, call ``agent.learn_step()``.
3. Evaluate and checkpoint at the configured cadence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import clip_action, get_env_spec, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average, set_all_seeds
from rl_from_scratch.dreamer.agent import DreamerAgent
from rl_from_scratch.dreamer.config import DreamerConfig
import rl_from_scratch.dreamer.reporting as _dreamer_reporting

logger = logging.getLogger("rl_from_scratch")


# ──────────────────────────────────────────────────────────────────────────────
# Greedy evaluation helper
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(
    agent: DreamerAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate the Dreamer agent deterministically on a fresh environment."""
    eval_env = make_env(env_id, seed=seed)
    rewards: list[float] = []
    lengths: list[int] = []

    # Snapshot the training recurrent state so evaluation (which advances the
    # acting state via select_action) does not disturb the ongoing training
    # episode, and never flushes the training buffer.
    saved_state = (agent._h, agent._z, agent._prev_action)
    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            # Fresh belief per eval episode — no buffer side-effect
            agent._reset_recurrent()
            total_reward = 0.0
            ep_length = 0

            for _ in range(max_steps):
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
        # Restore the training recurrent state; leave the buffer untouched
        agent._h, agent._z, agent._prev_action = saved_state
        eval_env.close()

    r = np.asarray(rewards, dtype=np.float32)
    summary: dict[str, float] = {
        "mean_reward": float(r.mean()) if len(r) > 0 else 0.0,
        "std_reward": float(r.std()) if len(r) > 0 else 0.0,
        "min_reward": float(r.min()) if len(r) > 0 else 0.0,
        "max_reward": float(r.max()) if len(r) > 0 else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    if solved_reward is not None:
        summary["success_rate"] = float(
            sum(1 for x in rewards if x >= solved_reward) / max(1, n_episodes)
        )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Training entry-point
# ──────────────────────────────────────────────────────────────────────────────


@register_agent("dreamer")
def train_dreamer(
    config: DreamerConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a DreamerV1 agent.

    Parameters
    ----------
    config:
        A ``DreamerConfig`` dataclass.
    output_dir:
        Override the config's ``output_dir``.
    run_name:
        Override the config's ``run_name``.
    seed:
        Override the config's ``seed``.
    render:
        Ignored (rendering not supported in the main loop).

    Returns
    -------
    dict
        ``{"agent": DreamerAgent, "history": dict, "metrics": dict,
           "paths": ExperimentPaths}``.
    """
    del render
    config = apply_overrides(
        config, DreamerConfig, seed=seed, run_name=run_name, output_dir=output_dir
    )

    set_all_seeds(config.seed)

    env = make_env(config.env_id, seed=config.seed)

    try:
        env.action_space.seed(config.seed)
        spec = get_env_spec(env)

        obs_dim = spec.observation.dim
        action_dim = spec.action.dim
        action_low = np.asarray(spec.action.low, dtype=np.float32)
        action_high = np.asarray(spec.action.high, dtype=np.float32)

        agent = build_agent(
            DreamerAgent,
            config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )

        def _evaluate_fn(
            ag: DreamerAgent,
            env_id: str,
            *,
            n_episodes: int,
            seed: int,
            max_steps: int,
            solved_reward: float | None = None,
        ) -> dict[str, float]:
            return evaluate(
                ag, env_id,
                n_episodes=n_episodes,
                seed=seed,
                max_steps=max_steps,
                solved_reward=solved_reward,
            )

        manager = RunManager.from_config(config, agent=agent, evaluate_fn=_evaluate_fn)

        # ── Warm-up: collect transitions with uniform random policy ────────
        logger.info(
            "Dreamer: collecting %d warm-up steps on %s",
            config.num_warmup_steps,
            config.env_id,
        )
        warmup_obs, _ = env.reset(seed=config.seed)
        warmup_collected = 0

        while warmup_collected < config.num_warmup_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(
                warmup_obs, action, float(reward), next_obs, bool(terminated)
            )
            warmup_collected += 1
            warmup_obs = next_obs
            if terminated or truncated:
                agent.episode_ended()
                warmup_obs, _ = env.reset(seed=config.seed + warmup_collected)

        # Seal any in-progress warm-up episode so it is not fused with the
        # first post-warm-up episode (which begins after an env reset).
        agent.episode_ended()

        # ── Main training loop ─────────────────────────────────────────────
        total_steps = config.epochs * config.steps_per_epoch
        logger.info(
            "Dreamer: starting %d epochs × %d steps = %d total steps on %s",
            config.epochs,
            config.steps_per_epoch,
            total_steps,
            config.env_id,
        )

        global_step = config.num_warmup_steps
        obs, _ = env.reset(seed=config.seed)
        episode_reward = 0.0
        episode_length = 0
        ep_idx = 0

        pbar = tqdm(range(config.epochs), desc="dreamer", mininterval=1.0)

        for _epoch in pbar:
            for step_in_epoch in range(config.steps_per_epoch):
                # Act
                action = agent.select_action(obs)
                env_action = np.asarray(clip_action(action, env), dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(env_action)

                # Store (use ``terminated`` as done for value bootstrap)
                agent.store_transition(
                    obs, env_action, float(reward), next_obs, bool(terminated)
                )
                episode_reward += float(reward)
                episode_length += 1
                global_step += 1
                obs = next_obs

                # Episode boundary
                if terminated or truncated:
                    manager.record_episode(reward=episode_reward, length=episode_length)
                    pbar.set_postfix(
                        ep_r=f"{episode_reward:.1f}",
                        avg=f"{moving_average(manager.history['episode_rewards'], 10):.1f}",
                    )
                    episode_reward = 0.0
                    episode_length = 0
                    ep_idx += 1
                    agent.episode_ended()
                    obs, _ = env.reset(seed=config.seed + global_step)

                # Learn
                if global_step % config.train_every == 0:
                    m = agent.learn_step()
                    if m:
                        manager.record_updates(m)

                # Periodic eval and checkpoint
                manager.maybe_eval(agent, episode=ep_idx, timestep=global_step)
                manager.maybe_checkpoint(step=global_step)

        # Record the last in-progress episode if it has any steps
        if episode_length > 0:
            manager.record_episode(reward=episode_reward, length=episode_length)

        return manager.finalize_run(
            agent,
            reporting_module=_dreamer_reporting,
            observed_timesteps=global_step,
        )

    finally:
        env.close()
