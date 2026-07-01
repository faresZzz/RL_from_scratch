"""Tests for env factory from rl_from_scratch.core.env."""

import numpy as np
import pytest

from rl_from_scratch.core.env import get_env_info, make_env


def test_make_env_creates_cartpole():
    env = make_env("CartPole-v1")
    obs, info = env.reset()
    assert obs is not None
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs is not None
    env.close()


def test_make_env_seeds_reproducibly():
    env_a = make_env("CartPole-v1", seed=42)
    env_b = make_env("CartPole-v1", seed=42)

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)

    np.testing.assert_array_equal(obs_a, obs_b)
    env_a.close()
    env_b.close()


def test_get_env_info_cartpole():
    env = make_env("CartPole-v1")
    info = get_env_info(env)
    assert info["obs_dim"] == 4
    assert info["action_dim"] == 2
    assert info["is_discrete"] is True
    env.close()
