"""Training loops for Dyna-Q, Dyna-Q+, and Deep Dyna.

The **Dyna** architecture interleaves real environment interaction with
simulated planning using a learned world model:

1. **Real step**: observe ``s``, take ``a``, receive ``r, s'``.
   Update Q(s,a) via the standard TD(0) update:

       Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

2. **Store** the transition ``(s, a, r, s')`` in the model (a lookup table
   for tabular Dyna, a neural network for Deep Dyna).

3. **Planning steps**: sample ``k`` previously seen transitions from the
   model and apply the same TD update on the simulated data.

**Dyna-Q+** adds an exploration bonus ``kappa * sqrt(tau)`` where ``tau``
is the number of steps since ``(s,a)`` was last visited.  This encourages
re-exploration of stale transitions when the environment changes.

**Deep Dyna** replaces the tabular model with a neural network that predicts
``(r, s')`` given ``(s, a)``, and uses a standard DQN-style TD loss for the
Q-network.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average, set_all_seeds
from rl_from_scratch.dyna.agent import DeepDynaAgent, DynaQAgent, DynaQPlusAgent
from rl_from_scratch.dyna.config import DeepDynaConfig, DynaQConfig, DynaQPlusConfig
import rl_from_scratch.core.reporting as _core_reporting
import rl_from_scratch.dyna.reporting as _dyna_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# CartPole discretizer (kept inside the dyna package)
# ======================================================================

class CartPoleDiscretizer:
    """Discretize the continuous CartPole observation space into bins.

    Clips extreme velocities to ``[cart_velocity_min, cart_velocity_max]``
    and ``[pole_angular_velocity_min, pole_angular_velocity_max]`` so that
    the bins cover the typical operating range.
    """

    def __init__(self, config: DynaQConfig | DynaQPlusConfig, observation_space: Any) -> None:
        low = np.asarray(observation_space.low, dtype=np.float64).copy()
        high = np.asarray(observation_space.high, dtype=np.float64).copy()
        low[1] = config.cart_velocity_min
        high[1] = config.cart_velocity_max
        low[3] = config.pole_angular_velocity_min
        high[3] = config.pole_angular_velocity_max
        self.bins = tuple(
            np.linspace(low[index], high[index], config.bins[index]) for index in range(4)
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(len(edges) for edges in self.bins)  # type: ignore[return-value]

    def transform(self, observation: Any) -> tuple[int, int, int, int]:
        values = np.asarray(observation, dtype=np.float64)
        indices: list[int] = []
        for value, edges in zip(values, self.bins):
            index = int(np.digitize(value, edges) - 1)
            indices.append(int(np.clip(index, 0, len(edges) - 1)))
        return tuple(indices)  # type: ignore[return-value]


# ======================================================================
# Episode functions
# ======================================================================

def train_one_tabular_episode(
    agent: DynaQAgent | DynaQPlusAgent,
    env: Any,
    *,
    discretizer: CartPoleDiscretizer,
    max_steps: int,
    seed: int | None,
    random_action: bool = False,
) -> dict[str, Any]:
    """Collect one tabular Dyna episode with interleaved planning.

    For each real step:
    1. Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s') - Q(s,a)]
    2. Store (s,a,r,s') in the model.
    3. Repeat ``planning_steps`` times: sample from model, apply TD update.
    """
    obs, _ = env.reset(seed=seed)
    state = discretizer.transform(obs)
    total_reward = 0.0
    updates: list[dict[str, float]] = []

    for step in range(max_steps):
        action = (
            int(agent.rng.integers(agent.action_count))
            if random_action
            else agent.select_action(state, deterministic=False)
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretizer.transform(next_obs)
        done = terminated or truncated
        total_reward += float(reward)

        # --- Real-experience TD update + model storage ---
        updates.append(
            agent.learn_real_transition(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=done,
            )
        )

        # --- Planning: k simulated updates from the model ---
        for _ in range(agent.planning_steps):
            updates.append(agent.planning_step())

        state = next_state
        if done:
            break

    agent.episode_ended()
    return {"reward": total_reward, "length": step + 1, "updates": updates}


def train_one_deep_dyna_episode(
    agent: DeepDynaAgent,
    env: Any,
    *,
    max_steps: int,
    seed: int | None,
) -> dict[str, Any]:
    """Collect one Deep Dyna episode with neural model-based planning.

    Each step: real Q-learning update + neural model update +
    ``planning_steps`` simulated transitions.
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    updates: list[dict[str, float]] = []

    for step in range(max_steps):
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        agent.store_transition(obs, action, float(reward), next_obs, done)
        updates.append(agent.learn_step())

        obs = next_obs
        if done:
            break

    agent.episode_ended()
    return {"reward": total_reward, "length": step + 1, "updates": updates}


# ======================================================================
# Dyna evaluation (needs discretizer for tabular variants)
# ======================================================================

def evaluate(
    agent: DynaQAgent | DynaQPlusAgent | DeepDynaAgent,
    *,
    env_id: str,
    max_steps: int,
    n_episodes: int,
    seed: int,
    discretizer: CartPoleDiscretizer | None = None,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Greedy evaluation for Dyna agents (tabular or deep)."""
    eval_env = gym.make(env_id)
    rewards: list[float] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            state = discretizer.transform(obs) if discretizer is not None else obs
            total_reward = 0.0

            for step in range(max_steps):
                action = agent.select_action(state, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                state = discretizer.transform(obs) if discretizer is not None else obs
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)
    finally:
        eval_env.close()

    r = np.asarray(rewards, dtype=np.float32)
    result: dict[str, float] = {
        "mean_reward": float(r.mean()) if len(r) else 0.0,
        "std_reward": float(r.std()) if len(r) else 0.0,
        "min_reward": float(r.min()) if len(r) else 0.0,
        "max_reward": float(r.max()) if len(r) else 0.0,
    }
    if solved_reward is not None:
        result["success_rate"] = float(sum(x >= solved_reward for x in rewards) / max(1, len(rewards)))
    return result


# ======================================================================
# Tabular Dyna training (Dyna-Q / Dyna-Q+)
# ======================================================================

def _train_tabular_dyna(
    *,
    config: DynaQConfig | DynaQPlusConfig,
    config_cls: type[DynaQConfig] | type[DynaQPlusConfig],
    agent_cls: type[DynaQAgent] | type[DynaQPlusAgent],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
) -> dict[str, Any]:
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    set_all_seeds(config.seed)
    env = make_env(config.env_id, seed=config.seed, render=False)

    try:
        rng = np.random.default_rng(config.seed)
        env.action_space.seed(config.seed)
        discretizer = CartPoleDiscretizer(config, env.observation_space)

        agent = build_agent(
            agent_cls, config,
            state_shape=discretizer.shape,
            action_count=int(env.action_space.n),
            rng=rng,
        )

        # Tabular dyna evaluate needs the discretizer — wrap it.
        def dyna_evaluate(agent, env_id, *, n_episodes, seed, max_steps, solved_reward=None):
            return evaluate(
                agent, env_id=env_id, max_steps=max_steps, n_episodes=n_episodes,
                seed=seed, discretizer=discretizer, solved_reward=solved_reward,
            )

        manager = RunManager.from_config(config, agent=agent, evaluate_fn=dyna_evaluate)

        episodes = config.episodes
        assert episodes is not None

        logger.info("Starting %s on %s (%d episodes)", config.approach, config.env_id, episodes)
        pbar = tqdm(range(1, episodes + 1), desc=config.approach, mininterval=1.0)

        for episode in pbar:
            # --- Collect one tabular episode with planning ---
            result = train_one_tabular_episode(
                agent, env,
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
                manager.record_updates(update_metrics)

            pbar.set_postfix(
                reward=f"{result['reward']:.1f}",
                avg=f"{moving_average(manager.history['episode_rewards'], window=20):.1f}",
                eps=f"{agent.epsilon:.3f}",
            )

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=episode)
            manager.maybe_eval(agent, episode=episode, timestep=episode)

        # Dyna tabular uses discretizer for greedy video — pass as record_greedy_fn.
        def _record_greedy(ag, cfg, run_dir, **kw):
            return _core_reporting.record_greedy_episode(
                ag, cfg, run_dir,
                discretizer=discretizer,
                video_dir=run_dir / "videos",
                name_prefix=f"{cfg.approach}_greedy",
            )

        return manager.finalize_run(
            agent,
            reporting_module=_dyna_reporting,
            extra_metrics={"final_epsilon": agent.epsilon},
            record_greedy_fn=_record_greedy,
        )
    finally:
        env.close()


# ======================================================================
# Deep Dyna training
# ======================================================================

def _train_deep_dyna(
    *,
    config: DeepDynaConfig,
    config_cls: type[DeepDynaConfig],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
) -> dict[str, Any]:
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    set_all_seeds(config.seed)
    env = make_env(config.env_id, seed=config.seed, render=False)

    try:
        rng = np.random.default_rng(config.seed)
        env.action_space.seed(config.seed)
        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        agent = build_agent(
            agent_cls=DeepDynaAgent, config=config,
            obs_dim=obs_dim, n_actions=n_actions, rng=rng,
        )
        manager = RunManager.from_config(config, agent=agent)

        episodes = config.episodes
        assert episodes is not None

        logger.info("Starting %s on %s (%d episodes)", config.approach, config.env_id, episodes)
        pbar = tqdm(range(1, episodes + 1), desc=config.approach, mininterval=1.0)

        for episode in pbar:
            # --- Collect one deep dyna episode ---
            result = train_one_deep_dyna_episode(
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
            reporting_module=_dyna_reporting,
            extra_metrics={"final_epsilon": agent.epsilon},
        )
    finally:
        env.close()


# ======================================================================
# Public entry-points
# ======================================================================

@register_agent("dyna_q")
def train_dyna_q(
    config: DynaQConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a Dyna-Q agent."""
    del render
    return _train_tabular_dyna(
        config=config, config_cls=DynaQConfig, agent_cls=DynaQAgent,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("dyna_q_plus")
def train_dyna_q_plus(
    config: DynaQPlusConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a Dyna-Q+ agent (with exploration bonus)."""
    del render
    return _train_tabular_dyna(
        config=config, config_cls=DynaQPlusConfig, agent_cls=DynaQPlusAgent,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("deep_dyna")
def train_deep_dyna(
    config: DeepDynaConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a Deep Dyna agent (neural world model)."""
    del render
    return _train_deep_dyna(
        config=config, config_cls=DeepDynaConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )
