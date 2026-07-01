"""Tabular training loops — Q-learning and SARSA.

Each function reads like the algorithm it implements:

**Q-learning** (off-policy):  at each step, choose action ε-greedy,
observe (r, s'), and update Q(s, a) toward ``r + γ·max_a' Q(s', a')``.

**SARSA** (on-policy):  at each step, choose action ε-greedy,
observe (r, s'), choose next action a' ε-greedy, and update Q(s, a)
toward ``r + γ·Q(s', a')``.

The only algorithmic difference is the bootstrap target — Q-learning uses
the greedy max, SARSA uses the action actually taken.  To keep the code
readable, the two algorithms have **separate** ``train_one_episode``
functions so a reader sees exactly one algorithm at a time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average
from rl_from_scratch.tabular.agent import QLearningAgent, SarsaAgent
from rl_from_scratch.tabular.config import QLearningConfig, SarsaConfig
from rl_from_scratch.tabular.discretization import CartPoleDiscretizer
import rl_from_scratch.tabular.reporting as _tabular_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# Evaluate (tabular-specific — needs discretizer)
# ======================================================================

def evaluate(
    agent: QLearningAgent | SarsaAgent,
    env_id: str,
    *,
    discretizer: CartPoleDiscretizer,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Run *n_episodes* greedy episodes on a discretized environment."""
    import gymnasium as gym

    eval_env = gym.make(env_id)
    rewards: list[float] = []
    lengths: list[int] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            state = discretizer.transform(obs)
            episode_reward = 0.0

            for step in range(max_steps):
                action = agent.select_action(state, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                state = discretizer.transform(obs)
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


# ======================================================================
# Q-learning episode  (off-policy)
# ======================================================================

def _train_one_episode_q_learning(
    agent: QLearningAgent,
    env: Any,
    *,
    discretizer: CartPoleDiscretizer,
    max_steps: int,
    seed: int | None,
    random_action: bool = False,
) -> dict[str, Any]:
    """Collect one episode of Q-learning experience.

    Q-learning update rule (applied after each transition):

        Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') − Q(s, a)]
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   off-policy bootstrap: greedy max over next state
    """
    obs, _ = env.reset(seed=seed)
    state = discretizer.transform(obs)
    total_reward = 0.0
    updates: list[dict[str, float]] = []

    for step in range(max_steps):
        # --- Action selection: ε-greedy (or random during warm-up) ---
        action = (
            agent.select_action_random()
            if random_action
            else agent.select_action(state, deterministic=False)
        )

        # --- Environment step ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretizer.transform(next_obs)
        # CartPole time-limit truncation is treated as terminal, matching the
        # gold-standard notebook 01 (standard CartPole simplification).
        done = terminated or truncated
        reward = float(reward)
        total_reward += reward

        # --- Q-learning update: bootstrap from max_a' Q(s', a') ---
        update = agent.learn_step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        updates.append(update)
        state = next_state

        if done:
            break

    agent.episode_ended()  # ε decay
    return {"reward": total_reward, "length": step + 1, "updates": updates}


# ======================================================================
# SARSA episode  (on-policy)
# ======================================================================

def _train_one_episode_sarsa(
    agent: SarsaAgent,
    env: Any,
    *,
    discretizer: CartPoleDiscretizer,
    max_steps: int,
    seed: int | None,
    random_action: bool = False,
) -> dict[str, Any]:
    """Collect one episode of SARSA experience.

    SARSA update rule (applied after each transition):

        Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]
                                          ^^^^^^^^^^
                                          on-policy bootstrap: action actually chosen

    Unlike Q-learning, SARSA selects the next action *before* the update
    (since the bootstrap depends on the action the policy would actually
    take), and carries that action forward to the next step.
    """
    obs, _ = env.reset(seed=seed)
    state = discretizer.transform(obs)
    total_reward = 0.0
    updates: list[dict[str, float]] = []

    # SARSA requires the next action before the update → pre-select a₀.
    action = (
        agent.select_action_random()
        if random_action
        else agent.select_action(state, deterministic=False)
    )

    for step in range(max_steps):
        # --- Environment step with the pre-selected action ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretizer.transform(next_obs)
        # CartPole time-limit truncation is treated as terminal, matching the
        # gold-standard notebook 01 (standard CartPole simplification).
        done = terminated or truncated
        reward = float(reward)
        total_reward += reward

        # Pre-select a' for the bootstrap (on-policy: same ε-greedy policy).
        next_action = 0 if done else agent.select_action(next_state, deterministic=False)

        # --- SARSA update: bootstrap from Q(s', a') ---
        update = agent.learn_step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            done=done,
        )
        updates.append(update)

        state = next_state
        action = next_action

        if done:
            break

    agent.episode_ended()  # ε decay
    return {"reward": total_reward, "length": step + 1, "updates": updates}


# ======================================================================
# Shared training driver (Q-learning or SARSA)
# ======================================================================

def _train_tabular(
    *,
    config: QLearningConfig | SarsaConfig,
    agent_cls: type[QLearningAgent] | type[SarsaAgent],
    config_cls: type[QLearningConfig] | type[SarsaConfig],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
    sarsa: bool = False,
) -> dict[str, Any]:
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    env = make_env(config.env_id, seed=config.seed, render=False)
    try:
        rng = np.random.default_rng(config.seed)
        env.action_space.seed(config.seed)

        discretizer = CartPoleDiscretizer.from_config(config, env.observation_space)
        agent = build_agent(
            agent_cls,
            config,
            state_shape=discretizer.shape,
            action_count=int(env.action_space.n),
            rng=rng,
        )

        # Tabular evaluate needs the discretizer → custom evaluate_fn.
        def tabular_evaluate(agent, env_id, *, n_episodes, seed, max_steps, solved_reward=None):
            return evaluate(
                agent, env_id,
                discretizer=discretizer,
                n_episodes=n_episodes,
                seed=seed,
                max_steps=max_steps,
                solved_reward=solved_reward,
            )

        manager = RunManager.from_config(config, agent=agent, evaluate_fn=tabular_evaluate)

        episodes = config.episodes
        assert episodes is not None

        episode_fn = _train_one_episode_sarsa if sarsa else _train_one_episode_q_learning

        logger.info("Starting %s on %s (%d episodes)", config.approach, config.env_id, episodes)
        pbar = tqdm(range(1, episodes + 1), desc=config.approach, mininterval=1.0)

        for episode in pbar:
            # --- Collect one episode (algorithm-specific) ---
            result = episode_fn(
                agent,
                env,
                discretizer=discretizer,
                max_steps=config.max_steps_per_episode,
                seed=config.seed + episode - 1,
                random_action=episode <= config.random_episodes,
            )

            # --- Record metrics ---
            manager.record_episode(
                reward=result["reward"],
                length=result["length"],
                epsilon=agent.epsilon,
            )
            for update_metrics in result["updates"]:
                if update_metrics:
                    manager.record_updates(update_metrics)

            pbar.set_postfix(
                reward=f"{result['reward']:.1f}",
                avg=f"{moving_average(manager.history['episode_rewards'], window=20):.1f}",
                eps=f"{agent.epsilon:.3f}",
            )

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=episode)
            manager.maybe_eval(agent, episode=episode, timestep=episode)

        return manager.finalize_run(
            agent,
            reporting_module=_tabular_reporting,
            extra_metrics={"final_epsilon": agent.epsilon},
        )
    finally:
        env.close()


# ======================================================================
# Public entry-points (registered for CLI dispatch)
# ======================================================================

@register_agent("q_learning")
def train_q_learning(
    config: QLearningConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a tabular Q-learning agent on a discretized environment."""
    del render
    return _train_tabular(
        config=config,
        agent_cls=QLearningAgent,
        config_cls=QLearningConfig,
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        sarsa=False,
    )


@register_agent("sarsa")
def train_sarsa(
    config: SarsaConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a tabular SARSA agent on a discretized environment."""
    del render
    return _train_tabular(
        config=config,
        agent_cls=SarsaAgent,
        config_cls=SarsaConfig,
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        sarsa=True,
    )
