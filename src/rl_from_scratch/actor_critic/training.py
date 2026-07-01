"""Training loops for A2C, A2C-GAE, and A3C.

All three algorithms are **on-policy actor-critic** methods that update the
policy using the advantage function A(s,a) = Q(s,a) − V(s):

    ∇J(θ) = 𝔼[∇log π_θ(a|s) · Â(s,a)]

where Â is estimated differently per variant:

- **A2C**: n-step advantage  Â_t = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) − V(s_t)
- **A2C-GAE**: Generalised Advantage Estimation
  Â_t = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}  where δ_t = r_t + γV(s_{t+1}) − V(s_t)
- **A3C**: asynchronous A2C-GAE across ``num_workers`` parallel processes
  sharing weights through POSIX shared memory.

The critic is updated to minimise the value loss:
    L_V = 𝔼[(V_φ(s) − G_t)²]

An entropy bonus ``H(π)`` encourages exploration:
    L_total = L_policy + c₁·L_V − c₂·H(π)

Note: ``_a3c_worker`` is at module level (not nested) so that
``torch.multiprocessing`` with the ``spawn`` start method can
serialise it correctly on macOS and Windows.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rl_from_scratch.core.artifacts import save_json
from rl_from_scratch.core.config import apply_overrides, build_agent, register_agent
from rl_from_scratch.core.env import clip_action, get_env_info, make_env
from rl_from_scratch.core.normalization import ObservationNormalizer
from rl_from_scratch.core.recording import RunManager
from rl_from_scratch.core.utils import set_all_seeds
from rl_from_scratch.actor_critic.agent import A2CAgent, A2CGAEAgent, A3CAgent
from rl_from_scratch.actor_critic.config import A2CConfig, A2CGAEConfig, A3CConfig
import rl_from_scratch.actor_critic.reporting as _ac_reporting

logger = logging.getLogger("rl_from_scratch")


# ======================================================================
# Helpers shared by A2C / A2C-GAE / A3C
# ======================================================================

def _normalize_observation(agent: A2CAgent, obs: Any, *, update: bool) -> Any:
    """Apply running-mean normalisation if the agent has an obs_normalizer."""
    normalizer = getattr(agent, "obs_normalizer", None)
    if normalizer is not None:
        return normalizer.normalize(np.asarray(obs, dtype=np.float32), update=update)
    return obs


def _record_update_metrics(manager: RunManager, metrics: dict[str, float]) -> None:
    """Record an update, aliasing total_loss -> loss for metric consistency."""
    payload = dict(metrics)
    if "total_loss" in payload and "loss" not in payload:
        payload["loss"] = payload["total_loss"]
    manager.record_updates(payload)


# ======================================================================
# A2C / A2C-GAE evaluation
# ======================================================================

def evaluate(
    agent: A2CAgent,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    max_steps: int,
    solved_reward: float | None = None,
) -> dict[str, float]:
    """Evaluate actor-critic with frozen normalisation stats, deterministic actions."""
    import gymnasium as gym

    eval_env = gym.make(env_id)
    rewards: list[float] = []

    try:
        for i in range(n_episodes):
            obs, _ = eval_env.reset(seed=seed + i)
            obs = _normalize_observation(agent, obs, update=False)
            total_reward = 0.0

            for step in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                env_action = np.asarray(clip_action(action, eval_env), dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = eval_env.step(env_action)
                obs = _normalize_observation(agent, next_obs, update=False)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)
    finally:
        eval_env.close()

    r = np.asarray(rewards, dtype=np.float32)
    summary: dict[str, float] = {
        "mean_reward": float(r.mean()) if len(r) else 0.0,
        "std_reward": float(r.std()) if len(r) else 0.0,
        "min_reward": float(r.min()) if len(r) else 0.0,
        "max_reward": float(r.max()) if len(r) else 0.0,
    }
    if solved_reward is not None:
        summary["success_rate"] = float(
            sum(1 for x in rewards if x >= solved_reward) / max(1, n_episodes)
        )
    return summary


# ======================================================================
# One A2C / A2C-GAE episode
# ======================================================================

def train_one_episode(
    agent: A2CAgent,
    env: Any,
    *,
    max_steps: int,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Collect one A2C/A2C-GAE episode, triggering updates when the buffer fills.

    The loop follows the on-policy actor-critic pattern:
      1. Sample a from pi_theta(.|s).
      2. Step environment, store transition.
      3. When the n-step buffer is full -> compute advantages (n-step TD
         or GAE) and run the actor-critic gradient step.

    Truncated (TimeLimit) episodes bootstrap V(s_{T+1}):
        r_adjusted = r + gamma * V(s')
    to avoid treating time-limits as terminal states.
    """
    obs, _ = env.reset(seed=seed)
    obs = _normalize_observation(agent, obs, update=True)
    total_reward = 0.0
    learn_results: list[dict[str, float]] = []
    terminated = False
    truncated = False

    for step in range(max_steps):
        # --- Sample action from pi_theta(a|s) ---
        selected_action = agent.select_action(obs, deterministic=False)
        policy_action = np.asarray(selected_action, dtype=np.float32)
        env_action = np.asarray(clip_action(policy_action, env), dtype=np.float32)

        # --- Environment step ---
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        next_obs = _normalize_observation(agent, next_obs, update=True)
        done = bool(terminated or truncated)
        env_reward = float(reward)
        stored_reward = env_reward

        # Bootstrap truncated episodes: r_adj = r + gamma * V(s')
        if truncated and not terminated:
            with torch.no_grad():
                terminal_value = float(agent.critic(agent._to_tensor(next_obs)).item())
            stored_reward = env_reward + agent.gamma * terminal_value

        agent.record_action_diagnostics(raw_action=policy_action, clipped_action=env_action)
        agent.store_transition(obs, policy_action, stored_reward, next_obs, done)
        total_reward += env_reward

        # --- Update when the n-step rollout buffer is full ---
        # Compute advantages (n-step TD or GAE) + policy/value gradient step.
        if agent.buffer.is_full():
            if done:
                next_value = 0.0
            else:
                with torch.no_grad():
                    next_value = float(agent.critic(agent._to_tensor(next_obs)).item())
            learn_result = agent.learn_step(next_value=next_value)
            if learn_result:
                learn_results.append(learn_result)

        obs = next_obs
        if done:
            break

    return {
        "reward": total_reward,
        "length": step + 1,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "seed": seed,
    }, learn_results


# ======================================================================
# A2C / A2C-GAE training driver
# ======================================================================

def _train_actor_critic(
    *,
    config: A2CConfig | A2CGAEConfig,
    agent_cls: type[A2CAgent],
    config_cls: type[A2CConfig] | type[A2CGAEConfig],
    output_dir: str | None,
    run_name: str | None,
    seed: int | None,
) -> dict[str, Any]:
    """Shared timestep-based training loop for A2C and A2C-GAE.

    Unlike REINFORCE (episode-based), updates happen every n_steps transitions
    -- not at the end of each episode.
    """
    config = apply_overrides(config, config_cls, seed=seed, run_name=run_name, output_dir=output_dir)

    set_all_seeds(config.seed)
    env = make_env(config.env_id, seed=config.seed, render=False)

    try:
        env.action_space.seed(config.seed)
        info = get_env_info(env)
        agent = build_agent(agent_cls, config, obs_dim=info["obs_dim"], action_dim=info["action_dim"])
        manager = RunManager.from_config(config, agent=agent, evaluate_fn=evaluate)

        global_step = 0
        episode_count = 0

        logger.info("Starting %s on %s (%d timesteps)", config.approach, config.env_id, config.total_timesteps)
        progress = tqdm(total=config.total_timesteps, desc=config.approach, mininterval=1.0)

        while global_step < config.total_timesteps:
            episode_count += 1

            # --- Collect one on-policy episode ---
            result, learn_results = train_one_episode(
                agent, env,
                max_steps=min(config.max_steps_per_episode, config.total_timesteps - global_step),
                seed=config.seed + episode_count - 1,
            )
            global_step += int(result["length"])
            progress.update(int(result["length"]))

            # --- Record metrics ---
            manager.record_episode(reward=float(result["reward"]), length=int(result["length"]))
            for lr in learn_results:
                _record_update_metrics(manager, lr)

            if episode_count % 10 == 0:
                progress.set_postfix(ep=episode_count, reward=f"{float(result['reward']):.1f}")

            # --- Periodic checkpoint & eval ---
            manager.maybe_checkpoint(step=global_step)
            manager.maybe_eval(agent, episode=episode_count, timestep=global_step)

        progress.close()
        return manager.finalize_run(agent, reporting_module=_ac_reporting, observed_timesteps=global_step)
    finally:
        env.close()


# ======================================================================
# A3C worker (must be at module level for multiprocessing spawn)
# ======================================================================

@dataclass(slots=True)
class WorkerRollout:
    """Summary of a rollout collected by one A3C worker."""
    next_obs: np.ndarray
    episode_reward: float
    episode_length: int
    episode_ended: bool
    completed_episode_reward: float | None
    completed_episode_length: int | None
    action_abs_mean: float
    action_clip_fraction: float
    steps_collected: int


def train_one_worker_rollout(
    *,
    local_actor: Any,
    local_critic: Any,
    env: Any,
    buffer: Any,
    obs: np.ndarray,
    episode_reward: float,
    episode_length: int,
    global_step_counter: Any,
    total_timesteps: int,
    gamma: float,
    max_steps_per_episode: int | None = None,
) -> WorkerRollout:
    """Collect up to ``t_max`` transitions for one A3C worker.

    Each worker runs its own environment copy and local actor/critic.
    Transitions are stored in a local rollout buffer; after collection,
    ``compute_worker_update`` computes gradients that will be pushed to the
    shared model.
    """
    action_abs_sum = 0.0
    action_clip_count = 0.0
    action_step_count = 0
    steps_collected = 0

    for _ in range(buffer.n_steps):
        with global_step_counter.get_lock():
            if global_step_counter.value >= total_timesteps:
                break
            global_step_counter.value += 1

        # --- Local actor forward pass (no gradient needed for rollout) ---
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = local_actor.get_distribution(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = local_critic(obs_t)

        policy_action = action.squeeze(0).numpy().astype(np.float32)
        env_action = np.asarray(clip_action(policy_action, env), dtype=np.float32)
        action_abs_sum += float(np.mean(np.abs(env_action)))
        action_clip_count += float(np.mean(np.abs(policy_action - env_action) > 1e-6))
        action_step_count += 1

        # --- Environment step ---
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        next_episode_length = episode_length + 1
        reached_time_limit = (
            max_steps_per_episode is not None
            and next_episode_length >= int(max_steps_per_episode)
        )
        truncated = bool(truncated or (reached_time_limit and not terminated))
        done = bool(terminated or truncated)
        env_reward = float(reward)
        stored_reward = env_reward

        # Bootstrap truncated (TimeLimit) episodes: r_adj = r + gamma * V(s')
        if truncated and not terminated:
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                terminal_value = float(local_critic(next_obs_t).item())
            stored_reward = env_reward + gamma * terminal_value

        buffer.push(
            np.asarray(obs, dtype=np.float32),
            policy_action,
            stored_reward,
            done,
            float(log_prob.item()),
            float(value.item()),
        )

        episode_reward += env_reward
        episode_length = next_episode_length
        steps_collected += 1

        if done:
            reset_obs, _ = env.reset()
            return WorkerRollout(
                next_obs=np.asarray(reset_obs, dtype=np.float32),
                episode_reward=0.0,
                episode_length=0,
                episode_ended=True,
                completed_episode_reward=episode_reward,
                completed_episode_length=episode_length,
                action_abs_mean=action_abs_sum / max(action_step_count, 1),
                action_clip_fraction=action_clip_count / max(action_step_count, 1),
                steps_collected=steps_collected,
            )

        obs = np.asarray(next_obs, dtype=np.float32)

    return WorkerRollout(
        next_obs=np.asarray(obs, dtype=np.float32),
        episode_reward=episode_reward,
        episode_length=episode_length,
        episode_ended=False,
        completed_episode_reward=None,
        completed_episode_length=None,
        action_abs_mean=action_abs_sum / max(action_step_count, 1),
        action_clip_fraction=action_clip_count / max(action_step_count, 1),
        steps_collected=steps_collected,
    )


def compute_worker_update(
    *,
    local_actor: Any,
    local_critic: Any,
    buffer: Any,
    rollout: WorkerRollout,
    gamma: float,
    gae_lambda: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Compute the A3C local update from a collected rollout.

    Steps:
    1. Compute GAE advantages and bootstrapped returns from the buffer.
    2. Normalise advantages (stabilises policy gradient).
    3. Compute losses:
       - Policy loss:  L_pi = -E[log pi(a|s) * A_hat]
       - Value loss:   L_V = MSE(V_phi(s), G_t)
       - Entropy bonus: H(pi)
       - Total: L = L_pi + c1*L_V - c2*H(pi)
    4. Backpropagate and clip gradients.
    """
    # Bootstrap value for the last state in the rollout.
    if rollout.episode_ended:
        next_value = 0.0
    else:
        next_obs_t = torch.as_tensor(rollout.next_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            next_value = float(local_critic(next_obs_t).item())

    # GAE: A_hat_t = sum (gamma*lambda)^k delta_{t+k}
    returns, advantages = buffer.compute_gae(next_value, gamma, gae_lambda)
    batch = buffer.get_batch()
    obs_b = batch["obs"]
    actions_b = batch["actions"]

    adv_mean = float(advantages.mean().item())
    adv_std = float(advantages.std().item())

    # Normalise advantages for gradient stability.
    advantages_std = advantages.std()
    if advantages_std > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)

    # Recompute log-probs and values under current parameters.
    dist = local_actor.get_distribution(obs_b)
    new_log_probs = dist.log_prob(actions_b).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1).mean()
    new_values = local_critic(obs_b)

    # --- Losses ---
    # Policy gradient: nabla J approx E[nabla log pi * A_hat]
    policy_loss = -(new_log_probs * advantages.detach()).mean()
    # Critic: fit V_phi(s) to bootstrapped returns G_t.
    value_loss = torch.nn.functional.mse_loss(new_values, returns.detach())
    # Entropy bonus encourages exploration.
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    local_actor.zero_grad()
    local_critic.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(local_actor.parameters()) + list(local_critic.parameters()),
        max_grad_norm,
    )

    # Explained variance: how well V(s) predicts G_t.
    with torch.no_grad():
        returns_var = returns.var(unbiased=False)
        if returns_var > 1e-8:
            explained_variance = 1.0 - (returns - new_values).var(unbiased=False) / (returns_var + 1e-8)
        else:
            explained_variance = torch.tensor(0.0)
        log_std = local_actor.log_std.detach()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "total_loss": float(total_loss.item()),
        "explained_variance": float(explained_variance.item()) if torch.is_tensor(explained_variance) else float(explained_variance),
        "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "log_std_mean": float(log_std.mean().item()),
        "log_std_min": float(log_std.min().item()),
        "log_std_max": float(log_std.max().item()),
        "action_abs_mean": float(rollout.action_abs_mean),
        "action_clip_fraction": float(rollout.action_clip_fraction),
    }


def _a3c_worker(
    rank: int,
    shared_actor: Any,
    shared_critic: Any,
    shared_optimizer: Any,
    config_dict: dict,
    global_step_counter: Any,
    global_episode_counter: Any,
    result_queue: Any,
) -> None:
    """A3C worker running in a subprocess.

    Worker loop:
    1. Sync local weights <- shared model.
    2. Collect t_max steps in the local environment.
    3. Compute GAE advantages and losses.
    4. Backprop on local models.
    5. Push gradients to shared model.
    6. Shared optimizer step.
    7. Send episode metrics to the main process queue.

    Imports are local to avoid serialisation issues with ``spawn``.
    """
    from rl_from_scratch.actor_critic.network import GaussianActor, CriticNetwork
    from rl_from_scratch.actor_critic.buffer import RolloutBuffer
    from rl_from_scratch.actor_critic.agent import A3CAgent
    from rl_from_scratch.actor_critic.config import A3CConfig
    from rl_from_scratch.core.env import make_env

    config = A3CConfig.from_dict(config_dict)
    seed = config.seed + rank

    env = make_env(config.env_id, seed=seed, render=False)
    try:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Local (non-shared) models -- gradients computed here then pushed.
        local_actor = GaussianActor(obs_dim, action_dim, config.hidden_dim)
        local_critic = CriticNetwork(obs_dim, config.hidden_dim)

        buffer = RolloutBuffer(config.t_max, obs_dim, action_dim)

        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        episode_length = 0

        while True:
            with global_step_counter.get_lock():
                current_step = global_step_counter.value
            if current_step >= config.total_timesteps:
                break

            # 1. Sync local <- shared.
            A3CAgent.sync_local_from_shared(local_actor, shared_actor)
            A3CAgent.sync_local_from_shared(local_critic, shared_critic)

            buffer.reset()

            # 2. Collect t_max transitions.
            rollout = train_one_worker_rollout(
                local_actor=local_actor,
                local_critic=local_critic,
                env=env,
                buffer=buffer,
                obs=np.asarray(obs, dtype=np.float32),
                episode_reward=episode_reward,
                episode_length=episode_length,
                global_step_counter=global_step_counter,
                total_timesteps=config.total_timesteps,
                gamma=config.gamma,
                max_steps_per_episode=config.max_steps_per_episode,
            )

            if rollout.steps_collected == 0:
                break

            obs = rollout.next_obs
            episode_reward = rollout.episode_reward
            episode_length = rollout.episode_length

            if rollout.completed_episode_reward is not None:
                with global_episode_counter.get_lock():
                    global_episode_counter.value += 1
                result_queue.put({
                    "type": "episode",
                    "reward": rollout.completed_episode_reward,
                    "length": rollout.completed_episode_length,
                })

            shared_optimizer.zero_grad()

            # 3. GAE + losses + gradient clipping.
            metrics = compute_worker_update(
                local_actor=local_actor,
                local_critic=local_critic,
                buffer=buffer,
                rollout=rollout,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
                max_grad_norm=config.max_grad_norm,
            )

            # 5. Push local gradients -> shared model.
            A3CAgent.push_gradients_to_shared(local_actor, shared_actor)
            A3CAgent.push_gradients_to_shared(local_critic, shared_critic)

            # 6. Shared optimizer step.
            shared_optimizer.step()

            # 7. Send metrics to main process.
            result_queue.put({"type": "loss", **metrics})
    finally:
        env.close()


# ======================================================================
# Public entry-points
# ======================================================================

@register_agent("a2c")
def train_a2c(
    config: A2CConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train an A2C agent (Advantage Actor-Critic)."""
    del render
    return _train_actor_critic(
        config=config, agent_cls=A2CAgent, config_cls=A2CConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("a2c_gae")
def train_a2c_gae(
    config: A2CGAEConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train an A2C-GAE agent (A2C with Generalised Advantage Estimation)."""
    del render
    return _train_actor_critic(
        config=config, agent_cls=A2CGAEAgent, config_cls=A2CGAEConfig,
        output_dir=output_dir, run_name=run_name, seed=seed,
    )


@register_agent("a3c")
def train_a3c(
    config: A3CConfig,
    *,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    render: bool | None = None,
) -> dict[str, Any]:
    """Train an A3C agent (Asynchronous Advantage Actor-Critic).

    Launches ``config.num_workers`` worker processes in parallel.  Each worker
    has its own environment copy and local actor/critic; weights are shared via
    POSIX shared memory.

    Uses ``mp.set_start_method("spawn")`` -- required on macOS.
    """
    del render
    mp.set_start_method("spawn", force=True)

    config = apply_overrides(config, A3CConfig, seed=seed, run_name=run_name, output_dir=output_dir)

    # Probe environment for obs_dim / action_dim.
    env_probe = make_env(config.env_id, seed=config.seed, render=False)
    info = get_env_info(env_probe)
    obs_dim: int = info["obs_dim"]
    action_dim: int = info["action_dim"]
    env_probe.close()

    agent = build_agent(A3CAgent, config, obs_dim=obs_dim, action_dim=action_dim)
    shared_actor, shared_critic, shared_optimizer = agent.create_shared_model()

    # Eval agent lives in the main process -- syncs from shared weights.
    eval_agent = A3CAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        lr=config.lr,
        gamma=config.gamma,
        n_steps=config.t_max,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        gae_lambda=config.gae_lambda,
        num_workers=config.num_workers,
        t_max=config.t_max,
    )
    manager = RunManager.from_config(config, agent=eval_agent, evaluate_fn=evaluate)

    # Shared counters.
    global_step_counter = mp.Value("i", 0)
    global_episode_counter = mp.Value("i", 0)
    result_queue: mp.Queue = mp.Queue()
    config_dict = config.to_dict()

    episode_count = 0

    # Launch worker processes.
    workers = []
    for rank in range(config.num_workers):
        p = mp.Process(
            target=_a3c_worker,
            args=(rank, shared_actor, shared_critic, shared_optimizer,
                  config_dict, global_step_counter, global_episode_counter, result_queue),
        )
        p.start()
        workers.append(p)

    _A3C_LOSS_KEYS = [
        "policy_loss", "value_loss", "entropy", "total_loss",
        "explained_variance", "grad_norm", "adv_mean", "adv_std",
        "log_std_mean", "log_std_min", "log_std_max",
        "action_abs_mean", "action_clip_fraction",
    ]

    progress = tqdm(total=config.total_timesteps, desc="a3c", mininterval=1.0)
    last_step = 0

    try:
        while any(p.is_alive() for p in workers):
            training_active = global_step_counter.value < config.total_timesteps

            try:
                item = result_queue.get(timeout=0.1)
                if item["type"] == "episode":
                    manager.record_episode(reward=item["reward"], length=item["length"])
                    episode_count += 1
                    if episode_count % 10 == 0:
                        progress.set_postfix(ep=episode_count, reward=f"{item['reward']:.1f}")
                elif item["type"] == "loss":
                    _record_update_metrics(
                        manager,
                        {k: float(item[k]) for k in _A3C_LOSS_KEYS if k in item},
                    )
            except Exception:
                pass

            # Update progress bar.
            new_step = global_step_counter.value
            if new_step > last_step:
                progress.update(new_step - last_step)
                last_step = new_step

            if training_active:
                current_step = global_step_counter.value
                # Sync eval agent weights for checkpoint/eval.
                if current_step >= manager._next_checkpoint_step or (
                    manager._next_eval_step is not None and current_step >= manager._next_eval_step
                ):
                    eval_agent.actor.load_state_dict(shared_actor.state_dict())
                    eval_agent.critic.load_state_dict(shared_critic.state_dict())
                manager.maybe_checkpoint(step=current_step)
                manager.maybe_eval(eval_agent, episode=episode_count, timestep=current_step)

        # Drain residual queue after workers exit.
        while not result_queue.empty():
            try:
                item = result_queue.get_nowait()
                if item["type"] == "episode":
                    manager.record_episode(reward=item["reward"], length=item["length"])
                    episode_count += 1
                elif item["type"] == "loss":
                    _record_update_metrics(
                        manager,
                        {k: float(item[k]) for k in _A3C_LOSS_KEYS if k in item},
                    )
            except Exception:
                break

        final_step = global_step_counter.value
        if final_step > last_step:
            progress.update(final_step - last_step)
        progress.close()

    finally:
        for p in workers:
            p.join()

    # Copy shared weights to the main agent.
    agent.actor.load_state_dict(shared_actor.state_dict())
    agent.critic.load_state_dict(shared_critic.state_dict())

    # Final checkpoint if no best was saved via eval.
    if manager.history["episode_rewards"] and not (manager.paths.checkpoint_dir / "best.pt").exists():
        manager.checkpoint(
            step=config.total_timesteps,
            agent=agent,
            keep_best=True,
            current_reward=float(np.max(manager.history["episode_rewards"])),
        )

    result = manager.finalize_run(
        agent,
        reporting_module=_ac_reporting,
        observed_timesteps=global_step_counter.value,
        extra_metrics={"num_workers": config.num_workers},
    )
    # Persist the extra metric.
    save_json(result["metrics"], manager.paths.metrics_path)
    return result
