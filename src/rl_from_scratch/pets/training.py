"""Training loop for PETS (Probabilistic Ensembles with Trajectory Sampling).

PETS (Chua et al. 2018) is an episode-based model-based RL algorithm.  Each
iteration follows the sequence:

1. **Ensemble fit**: train all ensemble members on the full collected buffer
   (with bootstrap resampling per member) using maximum-likelihood estimation.

2. **Real rollout**: collect one real episode using CEM planning at every step,
   storing transitions for the next iteration.

The loop is episode-based (not timestep-based): PETS extracts maximum
information from real data by re-fitting the full ensemble each iteration.

A random warm-up phase collects the first ``num_warmup_steps`` transitions
using uniform random actions before switching to CEM planning.
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
import rl_from_scratch.core.reporting as _core_reporting
from rl_from_scratch.pets.agent import PetsAgent
from rl_from_scratch.pets.config import PetsConfig
import rl_from_scratch.pets.reporting as _pets_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# Greedy evaluation
# ======================================================================

def evaluate(
    agent: PetsAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
    env_kwargs: dict | None = None,
) -> dict[str, float]:
    """Evaluate the PETS agent greedily on a fresh environment.

    During evaluation the agent still calls ``select_action`` (which uses CEM
    planning after warm-up).  The evaluation environment is separate so its
    transitions do not pollute the training buffer.

    Parameters
    ----------
    env_kwargs:
        Optional extra kwargs forwarded to ``make_env`` (e.g.
        ``exclude_current_positions_from_observation``).  Must match the
        kwargs used when building the training environment.
    """
    eval_env = make_env(env_id, seed=seed, env_kwargs=env_kwargs)
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

@register_agent("pets")
def train_pets(
    config: PetsConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a PETS agent.

    **Episode-based loop**: each iteration fits the ensemble on all collected
    data, then rolls out one real episode using CEM planning.

    Parameters
    ----------
    config:
        A ``PetsConfig`` dataclass.
    output_dir:
        Override the config's ``output_dir``.
    run_name:
        Override the config's ``run_name``.
    seed:
        Override the config's ``seed``.
    render:
        Ignored (rendering not supported in the episode loop).

    Returns
    -------
    dict
        ``{"agent": PetsAgent, "history": dict, "metrics": dict,
           "paths": ExperimentPaths}``.
    """
    del render
    config = apply_overrides(
        config, PetsConfig, seed=seed, run_name=run_name, output_dir=output_dir
    )

    set_all_seeds(config.seed)

    env = make_env(
        config.env_id,
        seed=config.seed,
        render=False,
        env_kwargs={"exclude_current_positions_from_observation": False},
    )

    try:
        env.action_space.seed(config.seed)
        spec = get_env_spec(env)

        obs_dim = spec.observation.dim
        action_dim = spec.action.dim
        action_low = np.asarray(spec.action.low, dtype=np.float32)
        action_high = np.asarray(spec.action.high, dtype=np.float32)

        # Retrieve environment timestep for the reward function
        dt = float(env.unwrapped.dt)

        # Build agent: build_agent pulls matching param names from config.
        # ``reward_dt`` is passed explicitly since it comes from the env,
        # not from the config dataclass.
        agent = build_agent(
            PetsAgent,
            config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            reward_dt=dt,
        )

        _env_kwargs = {"exclude_current_positions_from_observation": False}

        def _evaluate_fn(
            ag: PetsAgent,
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
                env_kwargs=_env_kwargs,
            )

        manager = RunManager.from_config(config, agent=agent, evaluate_fn=_evaluate_fn)

        # --- Warm-up: collect random transitions ---
        logger.info(
            "PETS: collecting %d warm-up steps on %s",
            config.num_warmup_steps,
            config.env_id,
        )
        warmup_collected = 0
        obs, _ = env.reset(seed=config.seed)

        while warmup_collected < config.num_warmup_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, float(reward), next_obs, bool(terminated or truncated))
            warmup_collected += 1
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset(seed=config.seed + warmup_collected)

        episodes = config.episodes
        assert episodes is not None and episodes > 0

        logger.info(
            "Starting PETS on %s (%d episodes)", config.env_id, episodes
        )
        pbar = tqdm(range(1, episodes + 1), desc="pets", mininterval=1.0)
        global_step = config.num_warmup_steps

        for episode in pbar:
            # --- 1. Fit ensemble on all collected data ---
            learn_metrics = agent.learn_step()

            # --- 2. Real rollout with CEM planning ---
            obs, _ = env.reset(seed=config.seed + episode)
            episode_reward = 0.0
            episode_length = 0

            for _ in range(config.max_steps_per_episode):
                action = agent.select_action(obs)
                env_action = np.asarray(clip_action(action, env), dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(env_action)
                done = bool(terminated or truncated)
                agent.store_transition(obs, env_action, float(reward), next_obs, done)
                episode_reward += float(reward)
                episode_length += 1
                global_step += 1
                obs = next_obs
                if done:
                    break

            agent.episode_ended()

            # --- 3. Record metrics ---
            manager.record_episode(reward=episode_reward, length=episode_length)
            manager.record_updates(learn_metrics)

            pbar.set_postfix(
                reward=f"{episode_reward:.1f}",
                avg=f"{moving_average(manager.history['episode_rewards'], window=5):.1f}",
                nll=f"{learn_metrics['dynamics_nll']:.3f}",
            )

            # --- 4. Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=episode)
            manager.maybe_eval(agent, episode=episode, timestep=global_step)

        def _record_greedy(ag: PetsAgent, cfg: PetsConfig, run_dir, **kw):
            return _core_reporting.record_greedy_episode(
                ag,
                cfg,
                run_dir,
                env_kwargs=_env_kwargs,
                **kw,
            )

        return manager.finalize_run(
            agent,
            reporting_module=_pets_reporting,
            record_greedy_fn=_record_greedy,
        )

    finally:
        env.close()
