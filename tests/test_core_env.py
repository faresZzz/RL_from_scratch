from __future__ import annotations

import numpy as np

from rl_from_scratch.core.env import clip_action, get_env_info, get_env_spec, make_env


def test_get_env_spec_cartpole_matches_legacy_info_contract() -> None:
    env = make_env("CartPole-v1")
    try:
        spec = get_env_spec(env)
        info = get_env_info(env)
    finally:
        env.close()

    assert spec.observation.shape == (4,)
    assert spec.observation.dim == 4
    assert spec.action.is_discrete is True
    assert spec.action.dim == 2
    assert spec.action.shape == ()
    assert spec.action.low is None
    assert spec.action.high is None

    assert info == {
        "obs_shape": (4,),
        "obs_dim": 4,
        "action_shape": (),
        "action_dim": 2,
        "is_discrete": True,
    }


def test_clip_action_respects_continuous_action_bounds() -> None:
    env = make_env("Pendulum-v1")
    try:
        clipped = clip_action(np.array([999.0], dtype=np.float32), env)
    finally:
        env.close()

    assert clipped.shape == (1,)
    assert float(clipped[0]) <= 2.0
    assert float(clipped[0]) >= -2.0
