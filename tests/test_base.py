"""Tests for BaseAgent and BaseConfig from rl_from_scratch.core.base."""

import pytest

from rl_from_scratch.core.base import BaseAgent, BaseConfig


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing default method behaviour
# ---------------------------------------------------------------------------

class _DummyAgent(BaseAgent):
    """Trivial concrete agent that implements abstract methods as no-ops."""

    def select_action(self, observation, *, deterministic: bool = False):
        return 0

    def learn_step(self, **kwargs):
        return {}

    def save(self, path):
        return path

    @classmethod
    def load(cls, path, **kwargs):
        return cls()


# ---------------------------------------------------------------------------
# BaseConfig tests
# ---------------------------------------------------------------------------


def test_base_config_default_values():
    config = BaseConfig()
    assert config.env_id == "CartPole-v1"
    assert config.gamma == pytest.approx(0.99)
    assert config.device == "auto"


def test_base_config_to_dict_roundtrip():
    original = BaseConfig()
    data = original.to_dict()
    restored = BaseConfig.from_dict(data)

    assert original.env_id == restored.env_id
    assert original.gamma == pytest.approx(restored.gamma)
    assert original.device == restored.device
    assert original.total_timesteps == restored.total_timesteps


def test_base_config_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        BaseConfig(gamma=1.5)
    with pytest.raises(ValueError):
        BaseConfig(gamma=-0.1)


def test_base_config_rejects_negative_timesteps():
    with pytest.raises(ValueError):
        BaseConfig(total_timesteps=0)
    with pytest.raises(ValueError):
        BaseConfig(total_timesteps=-100)


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


def test_base_agent_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseAgent()


def test_base_agent_store_transition_default_noop():
    agent = _DummyAgent()
    # Should not raise — default implementation is a no-op
    agent.store_transition(
        obs=0, action=0, reward=1.0, next_obs=1, done=False
    )


def test_base_agent_episode_ended_default_noop():
    agent = _DummyAgent()
    # Should not raise — default implementation is a no-op
    agent.episode_ended()
