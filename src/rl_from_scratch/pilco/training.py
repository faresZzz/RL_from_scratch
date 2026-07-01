"""Training loop for PILCO (Probabilistic Inference for Learning COntrol).

PILCO is an episode-based model-based reinforcement learning algorithm.
Each iteration follows the sequence:

1. **GP fit**: optimise hyperparameters of a GP dynamics model on all
   observed transitions (s, a) → Δs.

2. **Policy optimisation**: back-propagate through the analytic belief
   trajectory to minimise the expected saturating cost.

3. **Real rollout**: run the improved (deterministic) policy in the
   environment for one episode; store transitions for the next iteration.

The loop is deliberately episode-based (not timestep-based): PILCO is
sample-efficient precisely because it extracts maximum information from
every real rollout, so the natural iteration unit is the episode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.artifacts import save_json
from rl_from_scratch.core.env import NoEarlyTermination, clip_action, get_env_info, make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average, set_all_seeds
from rl_from_scratch.pilco.agent import DeepPilcoAgent, PilcoAgent
from rl_from_scratch.pilco.config import DeepPilcoConfig, PilcoConfig
import rl_from_scratch.pilco.reporting as _pilco_reporting

logger = logging.getLogger("rl_from_scratch")


def _sample_ip_reset_beliefs(
    agent: PilcoAgent,
    *,
    count: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    reset_variance = (0.02 ** 2) / 12.0
    beliefs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        raw = rng.uniform(low=-0.01, high=0.01, size=4)
        obs = torch.tensor(agent._encode(raw), dtype=torch.float64)
        sigma = torch.diag(
            torch.tensor(
                [reset_variance, reset_variance, 1e-8, reset_variance, reset_variance],
                dtype=torch.float64,
            )
        )
        beliefs.append((obs, sigma))
    return beliefs


def _sample_ip_reset_particles(
    agent: DeepPilcoAgent,
    env_id: str,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    env = make_env(env_id, seed=seed, render=False)
    samples: list[np.ndarray] = []
    try:
        for offset in range(count):
            obs, _ = env.reset(seed=seed + offset)
            samples.append(agent._encode(obs))
    finally:
        env.close()
    return torch.tensor(np.asarray(samples), dtype=agent.dtype)


def _copy_policy_state(agent: PilcoAgent | DeepPilcoAgent) -> dict[str, Any]:
    """Copy policy tensors so the best real-eval policy can be restored later."""
    return {
        key: value.detach().clone()
        for key, value in agent.policy.state_dict().items()
    }


def _restore_policy_state(
    agent: PilcoAgent | DeepPilcoAgent,
    state: dict[str, Any] | None,
) -> None:
    if state is not None:
        agent.policy.load_state_dict(state)


def _attach_final_eval(
    result: dict[str, Any],
    agent: PilcoAgent | DeepPilcoAgent,
    config: PilcoConfig | DeepPilcoConfig,
) -> dict[str, Any]:
    final_eval = evaluate(
        agent,
        config.env_id,
        n_episodes=config.final_eval_episodes,
        seed=config.final_eval_seed,
        max_steps=config.max_steps_per_episode,
        solved_reward=config.solved_reward,
    )
    result["metrics"]["final_eval_mean_reward"] = final_eval["mean_reward"]
    result["metrics"]["final_eval_min_reward"] = final_eval["min_reward"]
    result["metrics"]["final_eval_max_reward"] = final_eval["max_reward"]
    result["metrics"]["final_eval_std_reward"] = final_eval["std_reward"]
    result["metrics"]["final_eval_seed"] = float(config.final_eval_seed)
    result["metrics"]["final_eval_episodes"] = float(config.final_eval_episodes)
    result["metrics"]["selection_eval_seed"] = float(config.eval_seed)
    for name in ("mean_reward", "std_reward", "min_reward", "max_reward", "success_rate"):
        if name in final_eval:
            result["metrics"][f"heldout_eval_{name}"] = final_eval[name]
    save_json(result["metrics"], result["paths"].metrics_path)
    return result


# ======================================================================
# Greedy evaluation
# ======================================================================

def evaluate(
    agent: PilcoAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate the deterministic PILCO policy on a fresh environment.

    Uses the REAL task (with natural early termination) so balance duration
    reflects genuine policy quality.  The agent's ``select_action`` handles
    obs encoding internally when ``encode_angle=True``.
    """
    eval_env = make_env(env_id, seed=seed, render=False)
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            total_reward = 0.0
            ep_length = 0

            for _ in range(max_steps):
                # agent.select_action encodes obs internally when encode_angle=True
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

@register_agent("pilco")
def train_pilco(
    config: PilcoConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a PILCO agent.

    **Episode-based loop**: each iteration fits the GP, optimises the
    policy analytically, then collects one real rollout.
    """
    del render
    config = apply_overrides(
        config, PilcoConfig, seed=seed, run_name=run_name, output_dir=output_dir
    )

    set_all_seeds(config.seed)

    # Data-collection environment: optionally suppress early termination so
    # fixed-horizon rollouts include the pole falling (reference PILCO recipe).
    _base_env = make_env(config.env_id, seed=config.seed, render=False)
    if config.fixed_horizon_steps > 0:
        collect_env: Any = NoEarlyTermination(
            _base_env, fixed_horizon_steps=config.fixed_horizon_steps
        )
        logger.info(
            "PILCO: using NoEarlyTermination wrapper with fixed_horizon_steps=%d",
            config.fixed_horizon_steps,
        )
    else:
        collect_env = _base_env

    try:
        collect_env.action_space.seed(config.seed)
        info = get_env_info(collect_env)
        collect_steps = min(
            config.collection_horizon or config.max_steps_per_episode,
            config.max_steps_per_episode,
        )

        agent = build_agent(
            PilcoAgent,
            config,
            obs_dim=info["obs_dim"],
            action_dim=info["action_dim"],
            action_low=np.asarray(collect_env.action_space.low, dtype=np.float64),
            action_high=np.asarray(collect_env.action_space.high, dtype=np.float64),
        )
        if config.encode_angle and config.env_id == "InvertedPendulum-v5":
            agent.set_optimization_beliefs(
                _sample_ip_reset_beliefs(agent, count=config.n_initial_beliefs, seed=5000)
            )
        manager = RunManager.from_config(config, agent=agent, evaluate_fn=evaluate)

        # --- Initial random rollouts to seed the GP ---
        logger.info(
            "PILCO: collecting %d initial random rollout(s) on %s",
            config.num_init_rollouts,
            config.env_id,
        )
        for init_ep in range(config.num_init_rollouts):
            obs, _ = collect_env.reset(seed=config.seed + init_ep)
            # Record episode-initial ENCODED state for accurate μ₀/Σ₀ estimation.
            agent.buffer.start_episode(agent._encode(obs))
            for _ in range(collect_steps):
                action = collect_env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = collect_env.step(action)
                # A time-limit truncation is not a modelled failure state.
                agent.store_transition(obs, action, float(reward), next_obs, bool(terminated))
                obs = next_obs
                if terminated or truncated:
                    break

        episodes = config.episodes
        assert episodes is not None and episodes > 0

        logger.info(
            "Starting PILCO on %s (%d iterations)", config.env_id, episodes
        )
        pbar = tqdm(range(1, episodes + 1), desc="pilco", mininterval=1.0)
        best_policy_state: dict[str, Any] | None = None

        for iteration in pbar:
            # --- 1. Fit GP + optimise policy ---
            learn_metrics = agent.learn_step()

            # Decaying real-world exploration broadens the dynamics data early,
            # then lets later rollouts measure the learned feedback itself.
            agent.exploration_noise = config.exploration_noise * max(
                0.0,
                1.0 - (iteration - 1) / max(episodes - 1, 1),
            )

            # --- 2. Real rollout with improved policy ---
            obs, _ = collect_env.reset(
                seed=config.seed + config.num_init_rollouts + iteration - 1
            )
            # Record episode-initial ENCODED state for accurate μ₀/Σ₀ estimation.
            agent.buffer.start_episode(agent._encode(obs))
            episode_reward = 0.0
            episode_length = 0

            for _ in range(collect_steps):
                action = agent.select_action(obs, deterministic=False)
                env_action = np.asarray(
                    clip_action(action, collect_env), dtype=np.float32
                )
                next_obs, reward, terminated, truncated, _ = collect_env.step(env_action)
                done = bool(terminated or truncated)
                agent.store_transition(obs, env_action, float(reward), next_obs, bool(terminated))
                episode_reward += float(reward)
                episode_length += 1
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
                cost=f"{learn_metrics['predicted_cost']:.3f}",
            )

            # --- 4. Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=iteration)
            eval_result = manager.maybe_eval(agent, episode=iteration, timestep=iteration)
            if eval_result is not None and manager.best_eval_step == iteration:
                best_policy_state = _copy_policy_state(agent)
            elif eval_result is not None:
                _restore_policy_state(agent, best_policy_state)

        _restore_policy_state(agent, best_policy_state)
        result = manager.finalize_run(agent, reporting_module=_pilco_reporting)
        return _attach_final_eval(result, agent, config)

    finally:
        collect_env.close()


# ======================================================================
# Deep PILCO training entry-point
# ======================================================================

@register_agent("deep_pilco")
def train_deep_pilco(
    config: DeepPilcoConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a Deep PILCO agent (BNN dynamics + particle propagation).

    **Episode-based loop** mirroring ``train_pilco``:
    each iteration trains the BNN on buffered transitions, optimises the RBF
    policy through particle trajectory prediction, then collects one real rollout.
    """
    del render
    config = apply_overrides(
        config, DeepPilcoConfig, seed=seed, run_name=run_name, output_dir=output_dir
    )

    set_all_seeds(config.seed)

    # Data-collection environment: optionally suppress early termination so
    # fixed-horizon rollouts include the pole falling (reference PILCO recipe).
    _base_env = make_env(config.env_id, seed=config.seed, render=False)
    if config.fixed_horizon_steps > 0:
        collect_env: Any = NoEarlyTermination(
            _base_env, fixed_horizon_steps=config.fixed_horizon_steps
        )
        logger.info(
            "Deep PILCO: using NoEarlyTermination wrapper with fixed_horizon_steps=%d",
            config.fixed_horizon_steps,
        )
    else:
        # fixed_horizon_steps=0 keeps the env's natural early termination, which
        # mirrors the gold-standard notebook 10b. Deep PILCO is documented (REGLES
        # §7) as a hard method whose result is reported honestly, not forced via
        # the fixed-horizon recipe that classic PILCO needs.
        collect_env = _base_env

    # Evaluation uses the REAL task env (with natural termination) to measure
    # balance duration accurately — same as train_pilco.
    env = _base_env  # alias for legacy code below

    try:
        collect_env.action_space.seed(config.seed)
        info = get_env_info(collect_env)
        collect_steps = min(
            config.collection_horizon or config.max_steps_per_episode,
            config.max_steps_per_episode,
        )

        agent = build_agent(
            DeepPilcoAgent,
            config,
            obs_dim=info["obs_dim"],
            action_dim=info["action_dim"],
            action_low=np.asarray(collect_env.action_space.low, dtype=np.float32),
            action_high=np.asarray(collect_env.action_space.high, dtype=np.float32),
        )
        manager = RunManager.from_config(config, agent=agent, evaluate_fn=evaluate)

        # --- Initial random rollouts to seed the BNN ---
        logger.info(
            "Deep PILCO: collecting %d initial random rollout(s) on %s",
            config.num_init_rollouts,
            config.env_id,
        )
        for init_ep in range(config.num_init_rollouts):
            obs, _ = collect_env.reset(seed=config.seed + init_ep)
            # Record episode-initial ENCODED state for accurate μ₀/Σ₀ estimation.
            agent.buffer.start_episode(agent._encode(obs))
            for _ in range(collect_steps):
                action = collect_env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = collect_env.step(action)
                # A time-limit truncation is not a modelled failure state.
                agent.store_transition(obs, action, float(reward), next_obs, bool(terminated))
                obs = next_obs
                if terminated or truncated:
                    break

        episodes = config.episodes
        assert episodes is not None and episodes > 0

        logger.info(
            "Starting Deep PILCO on %s (%d iterations)", config.env_id, episodes
        )
        pbar = tqdm(range(1, episodes + 1), desc="deep_pilco", mininterval=1.0)
        best_policy_state: dict[str, Any] | None = None

        for iteration in pbar:
            # --- 1. Train BNN + optimise policy ---
            if config.encode_angle and config.env_id == "InvertedPendulum-v5":
                agent.set_reset_particles(
                    _sample_ip_reset_particles(agent, config.env_id, count=config.n_particles, seed=2000 + iteration)
                )
            learn_metrics = agent.learn_step(iteration=iteration)

            agent.exploration_noise = config.exploration_noise * max(
                0.0,
                1.0 - (iteration - 1) / max(episodes - 1, 1),
            )

            # --- 2. Real rollout with improved policy ---
            obs, _ = collect_env.reset(
                seed=config.seed + config.num_init_rollouts + iteration - 1
            )
            # Record episode-initial ENCODED state for accurate μ₀/Σ₀ estimation.
            agent.buffer.start_episode(agent._encode(obs))
            episode_reward = 0.0
            episode_length = 0

            for _ in range(collect_steps):
                action = agent.select_action(obs, deterministic=False)
                env_action = np.asarray(
                    clip_action(action, collect_env), dtype=np.float32
                )
                next_obs, reward, terminated, truncated, _ = collect_env.step(env_action)
                done = bool(terminated or truncated)
                agent.store_transition(obs, env_action, float(reward), next_obs, bool(terminated))
                episode_reward += float(reward)
                episode_length += 1
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
                cost=f"{learn_metrics['predicted_cost']:.3f}",
            )

            # --- 4. Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=iteration)
            eval_result = manager.maybe_eval(agent, episode=iteration, timestep=iteration)
            if eval_result is not None and manager.best_eval_step == iteration:
                best_policy_state = _copy_policy_state(agent)

        _restore_policy_state(agent, best_policy_state)
        result = manager.finalize_run(agent, reporting_module=_pilco_reporting)
        return _attach_final_eval(result, agent, config)

    finally:
        collect_env.close()
