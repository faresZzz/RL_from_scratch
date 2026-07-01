"""Generic greedy evaluation for neural-network agents.

Runs *n* deterministic episodes and returns summary statistics.  Works with
any agent that implements ``select_action(obs, deterministic=True)`` — i.e.
every ``BaseAgent`` subclass in the project.

Tabular agents that require a discretizer use their own ``evaluate()``
defined in ``tabular/training.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_from_scratch.core.env import clip_action, make_env


def evaluate(
    agent: Any,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Run *n_episodes* greedy episodes and return summary statistics.

    Parameters
    ----------
    agent:
        Trained agent with ``select_action(obs, deterministic=True)``.
    env_id:
        Gymnasium environment id used for evaluation.
    n_episodes:
        Number of evaluation episodes to run.
    seed:
        Base seed — episode *i* resets with ``seed + i``.
    max_steps:
        Maximum steps per episode (safety cap).
    solved_reward:
        If set, ``success_rate`` counts episodes ≥ this threshold.

    Returns
    -------
    dict
        Keys: ``mean_reward``, ``std_reward``, ``min_reward``,
        ``max_reward``, ``mean_length``, and optionally ``success_rate``.
    """
    eval_env = make_env(env_id, seed=seed, render=False)
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            episode_reward = 0.0

            for step in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                env_action = clip_action(action, eval_env)
                obs, reward, terminated, truncated, _ = eval_env.step(env_action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            lengths.append(step + 1)
    finally:
        eval_env.close()

    result: dict[str, float] = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
    }
    if solved_reward is not None:
        result["success_rate"] = sum(
            1 for r in rewards if r >= solved_reward
        ) / len(rewards)
    return result
