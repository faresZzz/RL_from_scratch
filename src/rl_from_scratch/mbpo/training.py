"""Training loop for MBPO (Model-Based Policy Optimization, Janner et al. 2019).

MBPO alternates between fitting a probabilistic ensemble that predicts
(Δstate, reward) and updating a SAC policy on a mix of real and imagined
transitions generated from the model.

Main loop (per epoch):
1. Fit the ensemble on the full env buffer.
2. For each step:
   a. Act in the real env with the SAC policy; store the transition.
   b. Every ``rollout_every`` steps, generate short imagined rollouts.
   c. Perform ``updates_per_step`` SAC updates on mixed batches.
3. Evaluate and checkpoint at the configured cadence.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import clip_action, get_env_spec, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average, set_all_seeds
from rl_from_scratch.mbpo.agent import MbpoAgent
from rl_from_scratch.mbpo.buffer import ModelBuffer
from rl_from_scratch.mbpo.config import MbpoConfig
import rl_from_scratch.mbpo.reporting as _mbpo_reporting

logger = logging.getLogger("rl_from_scratch")


def _effective_model_buffer_capacity(config: MbpoConfig) -> int:
    """Return the capacity implied by MBPO's model-retention schedule.

    MBPO is off-policy, so we do not clear imagined data after each model fit.
    But imagined transitions become stale when the dynamics ensemble changes.
    ``model_retain_epochs`` therefore caps the model buffer to roughly the
    number of rollouts generated over the last N epochs, with the explicit
    ``model_buffer_capacity`` remaining a hard upper bound.
    """
    rollouts_per_epoch = math.ceil(config.steps_per_epoch / config.rollout_every)
    retain_capacity = (
        config.model_retain_epochs
        * rollouts_per_epoch
        * config.rollout_batch_size
        * config.rollout_length
    )
    retain_capacity = max(config.sac_batch_size, retain_capacity)
    return max(1, min(config.model_buffer_capacity, retain_capacity))


# ======================================================================
# Greedy evaluation
# ======================================================================


def evaluate(
    agent: MbpoAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate the MBPO agent greedily on a fresh environment.

    Uses the SAC deterministic policy (tanh(μ) rescaled).
    """
    eval_env = make_env(env_id, seed=seed)
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
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


# ======================================================================
# Main training entry-point
# ======================================================================


@register_agent("mbpo")
def train_mbpo(
    config: MbpoConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train an MBPO agent.

    Parameters
    ----------
    config:
        An ``MbpoConfig`` dataclass.
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
        ``{"agent": MbpoAgent, "history": dict, "metrics": dict,
           "paths": ExperimentPaths}``.
    """
    del render
    config = apply_overrides(
        config, MbpoConfig, seed=seed, run_name=run_name, output_dir=output_dir
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
            MbpoAgent,
            config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )
        model_capacity = _effective_model_buffer_capacity(config)
        if model_capacity != config.model_buffer_capacity:
            agent.model_buffer = ModelBuffer(model_capacity)
            logger.info(
                "MBPO: model buffer capacity capped to %d by model_retain_epochs=%d",
                model_capacity,
                config.model_retain_epochs,
            )

        def _evaluate_fn(
            ag: MbpoAgent,
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

        # --- Warm-up: collect random transitions with uniform random policy ---
        logger.info(
            "MBPO: collecting %d warm-up steps on %s",
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
                warmup_obs, _ = env.reset(seed=config.seed + warmup_collected)

        total_steps = config.epochs * config.steps_per_epoch
        logger.info(
            "MBPO: starting %d epochs × %d steps = %d total steps on %s",
            config.epochs,
            config.steps_per_epoch,
            total_steps,
            config.env_id,
        )

        global_step = config.num_warmup_steps
        obs, _ = env.reset(seed=config.seed)
        episode_reward = 0.0
        episode_length = 0
        model_metrics: dict[str, float] = {}
        rollout_metrics: dict[str, float] = {}

        pbar = tqdm(range(config.epochs), desc="mbpo", mininterval=1.0)

        for epoch in pbar:
            # --- Fit ensemble on all real data at the start of each epoch ---
            model_metrics = agent.fit_model()

            for step_in_epoch in range(config.steps_per_epoch):
                # --- Act in real env ---
                action = agent.select_action(obs)
                env_action = np.asarray(clip_action(action, env), dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(env_action)

                # Store with terminated (not truncated) as done
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
                        nll=f"{model_metrics.get('model_nll', 0.0):.3f}",
                    )
                    episode_reward = 0.0
                    episode_length = 0
                    obs, _ = env.reset(seed=config.seed + global_step)

                # --- Model rollouts ---
                if global_step % config.rollout_every == 0:
                    rollout_metrics = agent.generate_model_rollouts(
                        config.rollout_length
                    )

                # --- SAC updates on mixed batch ---
                learn_metrics = agent.learn_step()

                # Record step-level update metrics
                combined_metrics: dict[str, float] = {}
                combined_metrics.update(model_metrics)
                combined_metrics.update(rollout_metrics)
                combined_metrics.update(learn_metrics)
                if combined_metrics:
                    manager.record_updates(combined_metrics)

                # Periodic timestep-based eval and checkpoint
                manager.maybe_eval(agent, episode=epoch, timestep=global_step)
                manager.maybe_checkpoint(step=global_step)

        # Record the last in-progress episode if it has any steps
        if episode_length > 0:
            manager.record_episode(reward=episode_reward, length=episode_length)

        return manager.finalize_run(
            agent,
            reporting_module=_mbpo_reporting,
            observed_timesteps=global_step,
        )

    finally:
        env.close()
