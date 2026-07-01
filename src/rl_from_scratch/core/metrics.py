"""Shared metric names and summary helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rl_from_scratch.core.utils import moving_average


def history_key_for_update_metric(name: str) -> str:
    """Map an update metric name to its flat public history key."""
    aliases = {
        "loss": "step_losses",
        "policy_loss": "step_policy_losses",
        "value_loss": "step_value_losses",
        "total_loss": "step_total_losses",
        "entropy": "step_entropies",
        "kl": "step_kl",
        "approx_kl": "step_approx_kls",
        "actor_loss": "step_actor_losses",
        "critic_loss": "step_critic_losses",
        "alpha_loss": "step_alpha_losses",
        "q_loss": "step_q_losses",
        "recon_loss": "step_recon_losses",
        "reward_loss": "step_reward_losses",
        "model_loss": "step_model_losses",
        "imagined_return": "step_imagined_returns",
        "model_prediction_loss": "step_model_prediction_losses",
        "latent_prediction_loss": "step_latent_prediction_losses",
        "rollout_prediction_loss": "step_rollout_prediction_losses",
        "representation_prediction_loss": "step_representation_prediction_losses",
        "representation_variance_loss": "step_representation_variance_losses",
        "representation_covariance_loss": "step_representation_covariance_losses",
        "reward_prediction_loss": "step_reward_prediction_losses",
        "continuation_loss": "step_continuation_losses",
        "done_prediction_loss": "step_done_prediction_losses",
        "variance_loss": "step_variance_losses",
        "covariance_loss": "step_covariance_losses",
        "latent_std": "step_latent_stds",
        "effective_rank": "step_effective_ranks",
        "collapse_gap": "step_collapse_gaps",
        "exploration_bonus": "step_exploration_bonuses",
    }
    return aliases.get(name, f"step_{name}s")


def append_update_metrics(history: dict[str, list[Any]], metrics: dict[str, float]) -> None:
    """Append one update worth of flat metrics into ``history``."""
    for name, value in metrics.items():
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        history.setdefault(history_key_for_update_metric(name), []).append(numeric_value)


def summarize_history(
    history: dict[str, list[Any]],
    *,
    total_timesteps: int | None = None,
    observed_timesteps: int | None = None,
    episodes_to_solve: int | None = None,
) -> dict[str, Any]:
    """Build the common run summary from the flat history contract."""
    episode_rewards = [float(v) for v in history.get("episode_rewards", [])]
    eval_mean_rewards = [float(v) for v in history.get("eval_mean_rewards", [])]
    eval_std_rewards = [float(v) for v in history.get("eval_std_rewards", [])]
    eval_min_rewards = [float(v) for v in history.get("eval_min_rewards", [])]
    eval_max_rewards = [float(v) for v in history.get("eval_max_rewards", [])]
    eval_steps = [int(v) for v in history.get("eval_steps", [])]
    eval_timesteps = [int(v) for v in history.get("eval_timesteps", [])]

    summary: dict[str, Any] = {
        "episodes": len(episode_rewards),
        "final_reward": episode_rewards[-1] if episode_rewards else 0.0,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_reward_last_20": moving_average(episode_rewards, window=20),
        "best_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "best_train_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "episodes_to_solve": episodes_to_solve,
    }
    if total_timesteps is not None:
        summary["total_timesteps"] = int(total_timesteps)
    if observed_timesteps is not None:
        summary["observed_timesteps"] = int(observed_timesteps)

    if eval_mean_rewards:
        best_eval_index = int(np.argmax(eval_mean_rewards))
        final_eval_index = len(eval_mean_rewards) - 1
        summary.update(
            {
                "best_eval_mean_reward": eval_mean_rewards[best_eval_index],
                "final_eval_mean_reward": eval_mean_rewards[final_eval_index],
            }
        )
        if len(eval_std_rewards) > best_eval_index:
            summary["best_eval_std_reward"] = eval_std_rewards[best_eval_index]
        if len(eval_min_rewards) > best_eval_index:
            summary["best_eval_min_reward"] = eval_min_rewards[best_eval_index]
        if len(eval_max_rewards) > best_eval_index:
            summary["best_eval_max_reward"] = eval_max_rewards[best_eval_index]
        if len(eval_steps) > best_eval_index:
            summary["best_eval_step"] = eval_steps[best_eval_index]
        if len(eval_timesteps) > best_eval_index:
            summary["best_eval_timestep"] = eval_timesteps[best_eval_index]

        if len(eval_std_rewards) > final_eval_index:
            summary["final_eval_std_reward"] = eval_std_rewards[final_eval_index]
        if len(eval_min_rewards) > final_eval_index:
            summary["final_eval_min_reward"] = eval_min_rewards[final_eval_index]
        if len(eval_max_rewards) > final_eval_index:
            summary["final_eval_max_reward"] = eval_max_rewards[final_eval_index]
        if len(eval_steps) > final_eval_index:
            summary["final_eval_step"] = eval_steps[final_eval_index]
        if len(eval_timesteps) > final_eval_index:
            summary["final_eval_timestep"] = eval_timesteps[final_eval_index]

    return summary
