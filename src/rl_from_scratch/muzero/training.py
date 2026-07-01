"""Readable MuZero training loop for discrete environments."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import get_env_spec, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average, set_all_seeds
from rl_from_scratch.muzero.agent import MuZeroAgent
from rl_from_scratch.muzero.config import MuZeroConfig
import rl_from_scratch.muzero.reporting as _muzero_reporting

logger = logging.getLogger("rl_from_scratch")


def evaluate(
    agent: MuZeroAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    env = make_env(env_id, seed=seed)
    rewards: list[float] = []
    lengths: list[int] = []
    acting_state = agent.snapshot_acting_state()
    try:
        for index in range(n_episodes):
            obs, _ = env.reset(seed=seed + index)
            total_reward = 0.0
            for step in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    lengths.append(step + 1)
                    break
            else:
                lengths.append(max_steps)
            rewards.append(total_reward)
    finally:
        agent.restore_acting_state(acting_state)
        env.close()

    reward_array = np.asarray(rewards, dtype=np.float32)
    result = {
        "mean_reward": float(reward_array.mean()) if len(reward_array) else 0.0,
        "std_reward": float(reward_array.std()) if len(reward_array) else 0.0,
        "min_reward": float(reward_array.min()) if len(reward_array) else 0.0,
        "max_reward": float(reward_array.max()) if len(reward_array) else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    if solved_reward is not None:
        result["success_rate"] = float(sum(r >= solved_reward for r in rewards) / max(1, len(rewards)))
    return result


def _play_episode(
    agent: MuZeroAgent,
    env: Any,
    *,
    seed: int,
    max_steps: int,
    random_policy: bool,
) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    length = 0

    for step in range(max_steps):
        if random_policy:
            action = int(env.action_space.sample())
            agent._pending_root_value = 0.0
            agent._pending_child_visits = np.full(agent.num_actions, 1.0 / agent.num_actions, dtype=np.float32)
            agent._pending_to_play = 1
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        agent.store_transition(obs, action, float(reward), next_obs, done)
        total_reward += float(reward)
        length = step + 1
        obs = next_obs
        if done:
            break

    if agent.current_game.actions:
        agent.episode_ended()
    return {"reward": total_reward, "length": length}


@register_agent("muzero")
def train_muzero(
    config: MuZeroConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    config = apply_overrides(
        config,
        MuZeroConfig,
        seed=seed,
        run_name=run_name,
        output_dir=output_dir,
    )

    set_all_seeds(config.seed)
    env = make_env(config.env_id, seed=config.seed, render=bool(render))
    try:
        spec = get_env_spec(env)
        if not spec.action.is_discrete:
            raise ValueError("MuZero package only supports discrete action spaces.")

        agent = build_agent(
            MuZeroAgent,
            config,
            obs_dim=spec.observation.dim,
            num_actions=spec.action.dim,
        )
        manager = RunManager.from_config(config, agent=agent, evaluate_fn=evaluate)

        total_env_steps = 0
        episode_index = 0

        for warmup_index in range(config.num_warmup_games):
            summary = _play_episode(
                agent,
                env,
                seed=config.seed + warmup_index,
                max_steps=config.max_steps_per_episode,
                random_policy=True,
            )
            total_env_steps += int(summary["length"])

        for iteration in range(config.training_steps):
            for episode_offset in range(config.selfplay_episodes_per_iteration):
                summary = _play_episode(
                    agent,
                    env,
                    seed=config.seed + config.num_warmup_games + episode_index,
                    max_steps=config.max_steps_per_episode,
                    random_policy=False,
                )
                total_env_steps += int(summary["length"])
                episode_index += 1
                manager.record_episode(reward=summary["reward"], length=int(summary["length"]))
                logger.info(
                    "MuZero episode %d reward %.2f avg10 %.2f",
                    episode_index,
                    summary["reward"],
                    moving_average(manager.history["episode_rewards"], window=10),
                )

            for _ in range(config.updates_per_iteration):
                manager.record_updates(agent.learn_step())

            manager.maybe_eval(agent, episode=episode_index, timestep=total_env_steps)
            manager.maybe_checkpoint(step=iteration + 1)

        return manager.finalize_run(
            agent,
            reporting_module=_muzero_reporting,
            observed_timesteps=total_env_steps,
        )
    finally:
        env.close()
