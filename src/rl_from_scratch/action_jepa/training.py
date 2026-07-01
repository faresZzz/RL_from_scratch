"""Training loop for Action-JEPA."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rl_from_scratch.action_jepa.agent import ActionJepaAgent
from rl_from_scratch.action_jepa.config import ActionJepaConfig
import rl_from_scratch.action_jepa.reporting as _action_jepa_reporting
from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import clip_action, get_env_spec, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average, set_all_seeds

logger = logging.getLogger("rl_from_scratch")


def evaluate(
    agent: ActionJepaAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate the planner greedily on a fresh environment."""
    del solved_reward
    eval_env = make_env(env_id, seed=seed)
    rewards: list[float] = []
    lengths: list[int] = []
    saved_mean = agent._prev_mean.clone() if agent._prev_mean is not None else None
    saved_planner_rng = agent.planner.get_rng_state()
    try:
        for episode in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + episode)
            agent._prev_mean = None
            total_reward = 0.0
            length = 0
            for _ in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(clip_action(action, eval_env))
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            lengths.append(length)
    finally:
        agent._prev_mean = saved_mean
        agent.planner.set_rng_state(saved_planner_rng)
        eval_env.close()

    reward_array = np.asarray(rewards, dtype=np.float32)
    return {
        "mean_reward": float(reward_array.mean()) if len(reward_array) else 0.0,
        "std_reward": float(reward_array.std()) if len(reward_array) else 0.0,
        "min_reward": float(reward_array.min()) if len(reward_array) else 0.0,
        "max_reward": float(reward_array.max()) if len(reward_array) else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def _collect_steps(
    env: Any,
    agent: ActionJepaAgent,
    *,
    num_steps: int,
    seed_offset: int,
    global_step: int,
) -> tuple[Any, int]:
    """Collect a short chunk of real experience with the current planner."""
    obs, _ = env.reset(seed=seed_offset)
    for step in range(num_steps):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(clip_action(action, env))
        agent.store_transition(obs, action, float(reward), next_obs, bool(terminated))
        global_step += 1
        obs = next_obs
        if terminated or truncated:
            agent.episode_ended()
            obs, _ = env.reset(seed=seed_offset + step + 1)
    agent.episode_ended()
    return obs, global_step


@register_agent("action_jepa")
def train_action_jepa(
    config: ActionJepaConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train Action-JEPA under the configured representation regime."""
    del render
    config = apply_overrides(
        config, ActionJepaConfig, seed=seed, run_name=run_name, output_dir=output_dir
    )
    set_all_seeds(config.seed)

    env = make_env(config.env_id, seed=config.seed)
    try:
        env.action_space.seed(config.seed)
        spec = get_env_spec(env)
        if spec.action.is_discrete or spec.action.low is None or spec.action.high is None:
            raise ValueError("Action-JEPA currently supports only continuous Box actions.")

        agent = build_agent(
            ActionJepaAgent,
            config,
            obs_dim=spec.observation.dim,
            action_dim=spec.action.dim,
            action_low=np.asarray(spec.action.low, dtype=np.float32),
            action_high=np.asarray(spec.action.high, dtype=np.float32),
        )
        manager = RunManager.from_config(config, agent=agent, evaluate_fn=evaluate)

        # Warm-up: collect random transitions so the first JEPA batches have real temporal structure.
        logger.info("Action-JEPA: collecting %d warm-up steps on %s", config.num_warmup_steps, config.env_id)
        obs, _ = env.reset(seed=config.seed)
        warmup_steps = 0
        while warmup_steps < config.num_warmup_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, float(reward), next_obs, bool(terminated))
            warmup_steps += 1
            obs = next_obs
            if terminated or truncated:
                agent.episode_ended()
                obs, _ = env.reset(seed=config.seed + warmup_steps)
        agent.episode_ended()
        global_step = warmup_steps

        # Pretraining alternates latent updates with fresh planner-collected data
        # (a light DAgger loop).  Stage-wise regimes use the Phase-A masked JEPA
        # objective, while joint keeps the historical end-to-end AC objective.
        logger.info(
            "Action-JEPA: starting %d pretraining updates (%s)",
            config.pretrain_steps,
            config.training_regime,
        )
        obs, _ = env.reset(seed=config.seed + global_step + 1)
        updates_done = 0
        if config.training_regime == "random-frozen":
            updates_done = config.pretrain_steps
        while updates_done < config.pretrain_steps:
            obs, global_step = _collect_steps(
                env,
                agent,
                num_steps=config.collect_every,
                seed_offset=config.seed + global_step + updates_done + 1,
                global_step=global_step,
            )
            updates_this_cycle = min(
                config.updates_per_collect,
                config.pretrain_steps - updates_done,
            )
            for _ in range(updates_this_cycle):
                if config.training_regime in {"stage-wise", "stage-wise-fair"}:
                    metrics = agent.representation_step()
                else:
                    metrics = agent.learn_step()
                manager.record_updates(metrics)
                updates_done += 1

        if config.freeze_encoder_after_pretrain:
            agent.freeze_encoder()

        # Control phase: run full episodes with the planner and keep refining the model online.
        logger.info("Action-JEPA: starting %d control episodes", config.episodes)
        obs, _ = env.reset(seed=config.seed + global_step + 10)
        episode_reward = 0.0
        episode_length = 0

        for episode in range(config.episodes):
            obs, _ = env.reset(seed=config.seed + global_step + episode + 10)
            agent._prev_mean = None
            episode_reward = 0.0
            episode_length = 0

            for _ in range(config.max_steps_per_episode):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(clip_action(action, env))
                agent.store_transition(obs, action, float(reward), next_obs, bool(terminated))
                metrics: dict[str, float] = {}
                for _ in range(config.control_updates_per_step):
                    metrics = agent.learn_step()
                    manager.record_updates(metrics)

                episode_reward += float(reward)
                episode_length += 1
                global_step += 1
                obs = next_obs
                if terminated or truncated:
                    break

            manager.record_episode(
                reward=episode_reward,
                length=episode_length,
                latent_std=metrics.get("latent_std", 0.0),
            )
            logger.info(
                "Action-JEPA episode %d/%d reward=%.2f avg10=%.2f",
                episode + 1,
                config.episodes,
                episode_reward,
                moving_average(manager.history["episode_rewards"], 10),
            )
            agent.episode_ended()
            manager.maybe_eval(agent, episode=episode + 1, timestep=global_step)
            manager.maybe_checkpoint(step=max(1, episode + 1))

        return manager.finalize_run(
            agent,
            reporting_module=_action_jepa_reporting,
            observed_timesteps=global_step,
        )
    finally:
        env.close()
