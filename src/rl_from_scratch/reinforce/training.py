"""Training loops for REINFORCE agents.

REINFORCE is a **Monte-Carlo policy gradient** method:

1. **Collect** a full episode trajectory τ = (s₀, a₀, r₁, s₁, a₁, r₂, …).
2. **Compute returns** Gₜ = Σ_{k=0}^{T-t} γ^k · r_{t+k+1} for each step.
3. **Update** the policy by ascending the gradient:

       ∇J(θ) ≈ (1/T) Σₜ ∇log π_θ(aₜ|sₜ) · Gₜ

   The **baseline variant** subtracts a learned state-value V(s) to reduce
   variance:

       ∇J(θ) ≈ (1/T) Σₜ ∇log π_θ(aₜ|sₜ) · (Gₜ − V_φ(sₜ))

The key property of REINFORCE is that it collects the **entire episode**
before any learning occurs — there is no replay buffer and no bootstrapping.
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import make_env
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import moving_average
from rl_from_scratch.reinforce.agent import ReinforceAgent, ReinforceBaselineAgent
from rl_from_scratch.reinforce.config import ReinforceBaselineConfig, ReinforceConfig
import rl_from_scratch.reinforce.reporting as _reinforce_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# One REINFORCE episode  (collect full trajectory → single update)
# ======================================================================

def train_one_episode(
    agent: ReinforceAgent | ReinforceBaselineAgent,
    env: Any,
    *,
    max_steps: int,
    seed: int | None,
) -> dict[str, Any]:
    """Collect a full episode, then run the REINFORCE policy-gradient update.

    Unlike DQN, REINFORCE stores the *entire* trajectory and updates once
    at the end.  ``agent.learn_step()`` computes the discounted returns
    Gₜ = Σ γ^k r_{t+k} and performs the policy gradient step.
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    for step in range(max_steps):
        # --- Sample action from the stochastic policy π_θ(a|s) ---
        action = agent.select_action(obs, deterministic=False)

        # --- Environment step ---
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = float(reward)
        total_reward += reward

        # --- Store transition (the full trajectory is needed for returns) ---
        agent.store_transition(obs, action, reward, next_obs, done)

        obs = next_obs
        if done:
            break

    # --- REINFORCE update: compute returns Gₜ, then ∇log π · Gₜ ---
    update_metrics = agent.learn_step()
    agent.episode_ended()
    return {"reward": total_reward, "length": step + 1, "updates": [update_metrics]}


# ======================================================================
# Shared training driver
# ======================================================================

def _train_reinforce(
    *,
    config: ReinforceConfig | ReinforceBaselineConfig,
    agent_cls: type[ReinforceAgent],
    config_cls: type[ReinforceConfig] | type[ReinforceBaselineConfig],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
) -> dict[str, Any]:
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    env = make_env(config.env_id, seed=config.seed, render=False)
    try:
        env.action_space.seed(config.seed)

        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)
        agent = build_agent(agent_cls, config, obs_dim=obs_dim, n_actions=n_actions)
        manager = RunManager.from_config(config, agent=agent)

        episodes = config.episodes
        assert episodes is not None

        logger.info("Starting %s on %s (%d episodes)", config.approach, config.env_id, episodes)
        pbar = tqdm(range(1, episodes + 1), desc=config.approach, mininterval=1.0)

        for episode in pbar:
            # --- Collect full episode + REINFORCE update ---
            result = train_one_episode(
                agent, env,
                max_steps=config.max_steps_per_episode,
                seed=config.seed + episode - 1,
            )

            # --- Record metrics ---
            manager.record_episode(reward=result["reward"], length=result["length"])
            for update_metrics in result["updates"]:
                if update_metrics:
                    manager.record_updates(update_metrics)

            pbar.set_postfix(
                reward=f"{result['reward']:.1f}",
                avg=f"{moving_average(manager.history['episode_rewards'], window=20):.1f}",
            )

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=episode)
            manager.maybe_eval(agent, episode=episode, timestep=episode)

        return manager.finalize_run(agent, reporting_module=_reinforce_reporting)
    finally:
        env.close()


# ======================================================================
# Public entry-points
# ======================================================================

@register_agent("reinforce")
def train_reinforce(
    config: ReinforceConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a REINFORCE agent (no baseline)."""
    del render
    return _train_reinforce(
        config=config, agent_cls=ReinforceAgent, config_cls=ReinforceConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("reinforce_baseline")
def train_reinforce_baseline(
    config: ReinforceBaselineConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train a REINFORCE agent with a learned state-value baseline."""
    del render
    return _train_reinforce(
        config=config, agent_cls=ReinforceBaselineAgent, config_cls=ReinforceBaselineConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )
