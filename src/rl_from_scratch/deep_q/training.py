"""Training loops for DQN-family agents (DQN, Double DQN, Rainbow DQN).

All three share the same episode-based off-policy loop:

1. **Collect**: at each step, choose action ε-greedy, step the environment,
   and store the transition ``(s, a, r, s', done)`` in the replay buffer.
2. **Learn**: sample a mini-batch from replay and minimise the TD loss.
   - **DQN**: ``L = (r + γ · max_{a'} Q_target(s', a') − Q(s, a))²``
   - **Double DQN**: ``a* = argmax_{a'} Q_online(s', a')``
     then ``L = (r + γ · Q_target(s', a*) − Q(s, a))²``
   - **Rainbow**: combines prioritised replay, dueling networks, n-step
     returns, and distributional Q-learning.
3. **Target sync**: periodically copy ``Q_online → Q_target``.

The replay buffer, target sync, and ε decay are managed inside the agent's
``learn_step()`` and ``episode_ended()`` — the training loop only sees the
high-level collect→learn cycle.
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
from rl_from_scratch.deep_q.agent import DQNAgent, DoubleDQNAgent, RainbowDQNAgent
from rl_from_scratch.deep_q.config import DQNConfig, DoubleDQNConfig, RainbowDQNConfig
import rl_from_scratch.deep_q.reporting as _deep_q_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# One episode of off-policy DQN collection + learning
# ======================================================================

def train_one_episode(
    agent: DQNAgent | DoubleDQNAgent | RainbowDQNAgent,
    env: Any,
    *,
    max_steps: int,
    seed: int | None,
) -> dict[str, Any]:
    """Collect one episode, learning from replay after every step.

    The inner loop is the standard off-policy DQN cycle:
    ``select_action → env.step → store_transition → learn_step``.
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    updates: list[dict[str, float]] = []

    for step in range(max_steps):
        # --- ε-greedy action selection ---
        action = agent.select_action(obs, deterministic=False)

        # --- Environment step ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # CartPole's time-limit truncation is treated as terminal, matching the
        # gold-standard notebook 02 (standard CartPole simplification). Kept
        # aligned with the notebook rather than splitting terminated/truncated.
        done = terminated or truncated
        reward = float(reward)
        total_reward += reward

        # --- Store transition in replay buffer ---
        agent.store_transition(obs, action, reward, next_obs, done)

        # --- Mini-batch gradient update from replay ---
        # Inside learn_step: sample batch, compute TD target, backprop,
        # optionally sync target network.
        update_metrics = agent.learn_step()
        if update_metrics:
            updates.append(update_metrics)

        obs = next_obs
        if done:
            break

    agent.episode_ended()  # ε decay + any end-of-episode bookkeeping
    return {"reward": total_reward, "length": step + 1, "updates": updates}


# ======================================================================
# Shared training driver for the DQN family
# ======================================================================

def _train_deep_q(
    *,
    config: DQNConfig | DoubleDQNConfig | RainbowDQNConfig,
    agent_cls: type[DQNAgent] | type[DoubleDQNAgent] | type[RainbowDQNAgent],
    config_cls: type[DQNConfig] | type[DoubleDQNConfig] | type[RainbowDQNConfig],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
) -> dict[str, Any]:
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    env = make_env(config.env_id, seed=config.seed, render=False)
    try:
        rng = np.random.default_rng(config.seed)
        env.action_space.seed(config.seed)

        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)
        agent = build_agent(agent_cls, config, obs_dim=obs_dim, n_actions=n_actions, rng=rng)
        manager = RunManager.from_config(config, agent=agent)

        episodes = config.episodes
        assert episodes is not None

        logger.info("Starting %s on %s (%d episodes)", config.approach, config.env_id, episodes)
        pbar = tqdm(range(1, episodes + 1), desc=config.approach, mininterval=1.0)

        for episode in pbar:
            # --- Collect one episode (off-policy DQN) ---
            result = train_one_episode(
                agent, env,
                max_steps=config.max_steps_per_episode,
                seed=config.seed + episode - 1,
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
            reporting_module=_deep_q_reporting,
            extra_metrics={"final_epsilon": agent.epsilon},
        )
    finally:
        env.close()


# ======================================================================
# Public entry-points (registered for CLI dispatch)
# ======================================================================

@register_agent("dqn")
def train_dqn(
    config: DQNConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a DQN agent on a discrete-action environment."""
    del render
    return _train_deep_q(
        config=config, agent_cls=DQNAgent, config_cls=DQNConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("double_dqn")
def train_double_dqn(
    config: DoubleDQNConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a Double DQN agent on a discrete-action environment."""
    del render
    return _train_deep_q(
        config=config, agent_cls=DoubleDQNAgent, config_cls=DoubleDQNConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("rainbow_dqn")
def train_rainbow_dqn(
    config: RainbowDQNConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a Rainbow DQN agent on a discrete-action environment."""
    del render
    return _train_deep_q(
        config=config, agent_cls=RainbowDQNAgent, config_cls=RainbowDQNConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )
