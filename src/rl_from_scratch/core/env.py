"""Shared environment helpers and specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


# ======================================================================
# Angle encoding helper (PILCO InvertedPendulum recipe)
# ======================================================================

def encode_obs(raw: np.ndarray) -> np.ndarray:
    """Encode a raw 4-D InvertedPendulum obs to the 5-D (sin/cos) representation.

    Raw observation layout (InvertedPendulum-v5):
        [cart_pos, pole_angle θ, cart_vel, pole_angvel]

    Encoded layout:
        [cart_pos, sin(θ), cos(θ), cart_vel, pole_angvel]

    Rationale: the angle θ is unbounded when the pole falls (wraps around).
    Replacing θ with (sin θ, cos θ) makes the representation periodic-safe
    and keeps the "upright" target at a fixed point: sin(0)=0, cos(0)=1.
    This is the standard recipe used by all reference PILCO implementations
    (nrontsis/PILCO, Ryan-Rhys/PILCO, aidanscannell/pilco-tensorflow) for
    fixed-horizon environments where the pole is allowed to fall.

    Parameters
    ----------
    raw:
        1-D array of shape ``(4,)`` in the order above.

    Returns
    -------
    np.ndarray of shape ``(5,)``.
    """
    raw = np.asarray(raw, dtype=np.float64)
    cart_pos = raw[0]
    theta = raw[1]
    cart_vel = raw[2]
    pole_angvel = raw[3]
    return np.array(
        [cart_pos, np.sin(theta), np.cos(theta), cart_vel, pole_angvel],
        dtype=np.float64,
    )


def project_encoded_angle_np(encoded: np.ndarray) -> np.ndarray:
    """Project encoded InvertedPendulum observations back to sin²+cos²=1.

    The encoded state layout is ``[x, sin(theta), cos(theta), x_dot, theta_dot]``.
    Model-based rollouts often learn/predict *deltas* in this encoded space. A
    plain residual update can drift to impossible pairs such as ``sin²+cos²=1.3``.
    Projection keeps imagined states on the physical angle manifold.
    """
    x = np.asarray(encoded).copy()
    sin_theta = x[..., 1]
    cos_theta = x[..., 2]
    norm = np.sqrt(sin_theta * sin_theta + cos_theta * cos_theta)
    norm = np.maximum(norm, 1e-8)
    x[..., 1] = sin_theta / norm
    x[..., 2] = cos_theta / norm
    return x


# ======================================================================
# Fixed-horizon environment wrapper
# ======================================================================

class NoEarlyTermination(gym.Wrapper):
    """Gymnasium wrapper that suppresses environment-initiated early termination.

    For PILCO data collection on InvertedPendulum-v5 the pole-fall ``terminated``
    flag cuts episodes to 5–20 steps when the policy is random/untrained, so the
    GP never observes the pole *falling* — it only sees the upright regime and
    believes "staying up is trivial".  This wrapper fixes that:

    - When ``terminated=True`` is returned by the base env (the pole fell past the
      task threshold ``|angle| > 0.2``), the wrapper IGNORES it and keeps stepping
      the *same* simulation. The MuJoCo physics past that threshold are perfectly
      valid, so the pole simply keeps falling and swinging — real dynamics. There
      is NO internal reset, so no fake "fallen -> freshly-upright" transition ever
      pollutes the GP training data.
    - The episode only ends when ``fixed_horizon_steps`` have been taken, or if a
      **genuine** failure occurs (non-finite observation, which would corrupt the
      GP).
    - The ``truncated`` flag (time-limit) signals the fixed horizon was reached so
      the training loop knows the episode is over. Reward and info are forwarded
      unchanged.

    This yields full fixed-horizon trajectories that include the pole falling and
    swinging through real continuous dynamics — the variety of data that PILCO's GP
    needs, without any spurious jumps.

    Usage (in training / data-collection only):
    ::

        env = NoEarlyTermination(gym.make("InvertedPendulum-v5"), fixed_horizon_steps=40)

    Parameters
    ----------
    env:
        The base Gymnasium environment.
    fixed_horizon_steps:
        Maximum steps per episode in this wrapper.  The training loop may
        impose its own cap via ``max_steps_per_episode`` as well — both caps
        apply (whichever is smaller).
    """

    def __init__(self, env: gym.Env, fixed_horizon_steps: int = 40) -> None:
        super().__init__(env)
        self.fixed_horizon_steps = int(fixed_horizon_steps)
        self._steps = 0
        self._last_seed: int | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:  # type: ignore[override]
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        # A genuine numerical failure (non-finite obs) must still end the episode,
        # otherwise it would corrupt the GP.
        if not np.isfinite(np.asarray(obs)).all():
            return obs, reward, True, truncated, info

        # Otherwise ignore the pole-fall `terminated`: keep stepping the SAME
        # simulation so the GP observes the real continued dynamics (the pole
        # falling and swinging). The sin/cos encoding keeps the unbounded angle
        # safe, and there is NO reset, hence no fake "fallen -> upright" transition.
        terminated = False

        # Hard cap: signal truncation once we reach the fixed horizon.
        if self._steps >= self.fixed_horizon_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, ...]
    dim: int


@dataclass(frozen=True)
class ActionSpec:
    shape: tuple[int, ...]
    dim: int
    is_discrete: bool
    low: np.ndarray | None = None
    high: np.ndarray | None = None


@dataclass(frozen=True)
class EnvSpec:
    env_id: str | None
    observation: ObservationSpec
    action: ActionSpec


def make_env(
    env_id: str,
    seed: int = 0,
    render: bool = False,
    record_video: bool = False,
    video_dir: str | None = None,
    record_every: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    """Create, seed, and optionally wrap a Gymnasium environment.

    Parameters
    ----------
    env_id:
        Gymnasium environment identifier.
    seed:
        Seed passed to ``env.reset`` and ``env.action_space.seed``.
    render:
        If True, creates the environment with ``render_mode="human"``.
    record_video:
        If True, wraps the environment with ``RecordVideo``.
    video_dir:
        Directory for recorded videos (required when ``record_video=True``).
    record_every:
        If provided, records every Nth episode; otherwise records the first.
    env_kwargs:
        Optional keyword arguments forwarded directly to ``gym.make``.
        Useful for environment-specific options such as
        ``exclude_current_positions_from_observation`` in MuJoCo envs.
        Defaults to ``None`` (no extra kwargs).
    """
    render_mode: str | None = "human" if render else None
    if record_video and not render_mode:
        render_mode = "rgb_array"

    extra_kwargs: dict[str, Any] = env_kwargs or {}
    env = gym.make(env_id, render_mode=render_mode, **extra_kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    if record_video:
        if video_dir is None:
            raise ValueError("video_dir is required when record_video is True.")
        trigger = (
            (lambda ep: ep % record_every == 0)
            if record_every is not None
            else None
        )
        kwargs: dict[str, Any] = {"video_folder": video_dir}
        if trigger is not None:
            kwargs["episode_trigger"] = trigger
        env = gym.wrappers.RecordVideo(env, **kwargs)

    return env


def get_env_spec(env: gym.Env) -> EnvSpec:
    """Describe the observation/action interface of an environment."""
    obs_space = env.observation_space
    act_space = env.action_space

    obs_shape = tuple(obs_space.shape) if obs_space.shape else ()
    obs_dim = int(obs_shape[0]) if obs_shape else 0

    is_discrete = isinstance(act_space, gym.spaces.Discrete)
    if is_discrete:
        action_shape: tuple[int, ...] = ()
        action_dim = int(act_space.n)
        low = None
        high = None
    else:
        action_shape = tuple(act_space.shape) if act_space.shape else ()
        action_dim = int(action_shape[0]) if action_shape else 0
        low = np.asarray(act_space.low, dtype=np.float32)
        high = np.asarray(act_space.high, dtype=np.float32)

    return EnvSpec(
        env_id=getattr(getattr(env, "spec", None), "id", None),
        observation=ObservationSpec(shape=obs_shape, dim=obs_dim),
        action=ActionSpec(
            shape=action_shape,
            dim=action_dim,
            is_discrete=is_discrete,
            low=low,
            high=high,
        ),
    )


def get_env_info(env: gym.Env) -> dict[str, Any]:
    """Legacy dict contract used across the current training code."""
    spec = get_env_spec(env)
    return {
        "obs_shape": spec.observation.shape,
        "obs_dim": spec.observation.dim,
        "action_shape": spec.action.shape,
        "action_dim": spec.action.dim,
        "is_discrete": spec.action.is_discrete,
    }


def clip_action(action: Any, env_or_spec: gym.Env | EnvSpec) -> Any:
    """Clip a continuous action to the environment bounds."""
    spec = env_or_spec if isinstance(env_or_spec, EnvSpec) else get_env_spec(env_or_spec)
    if spec.action.is_discrete or spec.action.low is None or spec.action.high is None:
        return action
    return np.clip(np.asarray(action, dtype=np.float32), spec.action.low, spec.action.high)
