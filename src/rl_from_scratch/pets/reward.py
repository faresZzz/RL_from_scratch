"""Reward functions for PETS.

PETS requires a differentiable (or at least callable) reward function for
planning; the agent cannot query the real environment inside the CEM rollout.
This module provides hand-coded reward functions that match the gymnasium
implementations, keyed by environment ID.

HalfCheetah-v5 reward decomposition (matches gymnasium source):
    forward_reward = (x_pos_after - x_pos_before) / dt
    ctrl_cost = ctrl_cost_weight * sum(action^2)
    total_reward = forward_reward_weight * forward_reward - ctrl_cost

where x_pos corresponds to ``obs[0]`` (x_velocity in standard obs) when
``exclude_current_positions_from_observation=False``.

Note: HalfCheetah-v5 with ``exclude_current_positions_from_observation=False``
includes the x-position as obs[0].  The forward reward is computed from the
*velocity* (obs[0] in the standard, position-excluded observation), but when
positions are included, obs[0] = x_pos and obs[8] = x_velocity.  We use the
position-delta formulation to match the env reward precisely.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def halfcheetah_reward(
    obs: Tensor,
    act: Tensor,
    next_obs: Tensor,
    *,
    dt: float,
    forward_reward_weight: float = 1.0,
    ctrl_cost_weight: float = 0.1,
) -> Tensor:
    """HalfCheetah-v5 reward function (works on arbitrary leading batch dims).

    Computes:
        forward_reward = (next_obs[..., 0] - obs[..., 0]) / dt
        ctrl_cost = ctrl_cost_weight * (act ** 2).sum(-1)
        reward = forward_reward_weight * forward_reward - ctrl_cost

    This matches the HalfCheetah gymnasium reward when the environment is
    created with ``exclude_current_positions_from_observation=False`` (so
    obs[0] is the x root position).

    Parameters
    ----------
    obs:
        Current observations, arbitrary leading dims, last dim = obs_dim.
    act:
        Actions, arbitrary leading dims, last dim = act_dim.
    next_obs:
        Next observations (same shape as obs).
    dt:
        Environment timestep (``env.unwrapped.dt``).
    forward_reward_weight:
        Scaling factor for the forward-velocity reward.
    ctrl_cost_weight:
        Scaling factor for the control-effort cost.

    Returns
    -------
    Tensor
        Scalar rewards, shape matching the leading dims of obs/act.
    """
    x_velocity = (next_obs[..., 0] - obs[..., 0]) / dt
    forward_reward = forward_reward_weight * x_velocity
    ctrl_cost = ctrl_cost_weight * (act ** 2).sum(dim=-1)
    return forward_reward - ctrl_cost


# Registry of available reward functions keyed by environment ID.
REWARD_FNS: dict[str, Callable[..., Tensor]] = {
    "HalfCheetah-v5": halfcheetah_reward,
}


def get_reward_fn(env_id: str) -> Callable[..., Tensor]:
    """Look up the reward function for *env_id*.

    Parameters
    ----------
    env_id:
        Gymnasium environment identifier.

    Returns
    -------
    Callable
        A reward function with signature
        ``(obs, act, next_obs, *, dt, ...) -> Tensor``.

    Raises
    ------
    ValueError
        If *env_id* is not in ``REWARD_FNS``.
    """
    if env_id not in REWARD_FNS:
        supported = ", ".join(sorted(REWARD_FNS))
        raise ValueError(
            f"No reward function registered for env '{env_id}'. "
            f"Supported: {supported}."
        )
    return REWARD_FNS[env_id]
