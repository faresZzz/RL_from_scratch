"""Tests for Rainbow DQN components.

These tests define the expected public API for the Rainbow DQN stack before
implementation lands.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest
import torch

from rl_from_scratch.core.config import load_config
from rl_from_scratch.deep_q.agent import RainbowDQNAgent
from rl_from_scratch.deep_q.buffer import (
    NStepTransitionAccumulator,
    PrioritizedReplayBuffer,
)
from rl_from_scratch.deep_q.config import RainbowDQNConfig
from rl_from_scratch.deep_q.network import (
    CategoricalDuelingQNetwork,
    DuelingQNetwork,
    NoisyLinear,
)
from rl_from_scratch.deep_q.training import train_rainbow_dqn


OBS_DIM = 4
N_ACTIONS = 2
HIDDEN_DIM = 16
BATCH_SIZE = 4
BUFFER_CAPACITY = 100
N_ATOMS = 11


def _rainbow_config(**overrides) -> RainbowDQNConfig:
    kwargs = {
        "env_id": "CartPole-v1",
        "episodes": 3,
        "checkpoint_every": 3,
        "device": "cpu",
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
        "buffer_capacity": BUFFER_CAPACITY,
        "n_atoms": N_ATOMS,
    }
    kwargs.update(overrides)
    return RainbowDQNConfig(**kwargs)


def _random_transition(obs_dim: int = OBS_DIM) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
    state = np.random.randn(obs_dim).astype(np.float32)
    action = int(np.random.randint(N_ACTIONS))
    reward = float(np.random.randn())
    next_state = np.random.randn(obs_dim).astype(np.float32)
    done = bool(np.random.rand() > 0.8)
    return state, action, reward, next_state, done


def _fill_agent_buffer(agent: RainbowDQNAgent, count: int = 32) -> None:
    for _ in range(count):
        agent.store_transition(*_random_transition())


def test_rainbow_config_defaults_and_yaml_dispatch(tmp_path) -> None:
    config = RainbowDQNConfig(episodes=3, checkpoint_every=3)
    assert config.approach == "rainbow_dqn"
    assert config.env_id == "CartPole-v1"

    yaml_path = tmp_path / "rainbow.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            approach: rainbow_dqn
            episodes: 3
            checkpoint_every: 3
            """
        ),
        encoding="utf-8",
    )

    loaded = load_config(yaml_path)
    assert isinstance(loaded, RainbowDQNConfig)
    assert loaded.approach == "rainbow_dqn"
    assert loaded.env_id == "CartPole-v1"


def test_noisy_linear_resample_changes_train_output_and_eval_is_stable() -> None:
    layer = NoisyLinear(OBS_DIM, HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, OBS_DIM)

    layer.train()
    out_before = layer(x)
    layer.resample_noise()
    out_after = layer(x)
    assert not torch.allclose(out_before, out_after)

    layer.eval()
    out_eval_1 = layer(x)
    out_eval_2 = layer(x)
    assert torch.allclose(out_eval_1, out_eval_2)


def test_dueling_q_network_output_shape() -> None:
    net = DuelingQNetwork(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
    )
    x = torch.randn(BATCH_SIZE, OBS_DIM)
    out = net(x)
    assert out.shape == (BATCH_SIZE, N_ACTIONS)


def test_categorical_dueling_q_network_outputs_normalized_probabilities() -> None:
    net = CategoricalDuelingQNetwork(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        n_atoms=N_ATOMS,
    )
    x = torch.randn(BATCH_SIZE, OBS_DIM)
    probs = net(x)

    assert probs.shape == (BATCH_SIZE, N_ACTIONS, N_ATOMS)
    assert torch.allclose(
        probs.sum(dim=-1),
        torch.ones(BATCH_SIZE, N_ACTIONS),
        atol=1e-5,
    )


def test_prioritized_replay_buffer_sample_shapes_and_priority_updates() -> None:
    buffer = PrioritizedReplayBuffer(capacity=BUFFER_CAPACITY, alpha=0.6, beta=0.4)

    for _ in range(16):
        buffer.push(*_random_transition())

    assert len(buffer) == 16

    sample = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones, indices, weights = sample

    assert states.shape == (BATCH_SIZE, OBS_DIM)
    assert actions.shape == (BATCH_SIZE,)
    assert rewards.shape == (BATCH_SIZE,)
    assert next_states.shape == (BATCH_SIZE, OBS_DIM)
    assert dones.shape == (BATCH_SIZE,)
    assert indices.shape == (BATCH_SIZE,)
    assert weights.shape == (BATCH_SIZE,)

    assert states.dtype == torch.float32
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.float32
    assert weights.dtype == torch.float32

    buffer.update_priorities(indices, np.linspace(0.5, 1.5, BATCH_SIZE, dtype=np.float32))


def test_n_step_transition_accumulator_aggregates_rewards_and_flushes_on_done() -> None:
    accumulator = NStepTransitionAccumulator(n_steps=3, gamma=0.9)

    first = accumulator.push("s0", 0, 1.0, "s1", False)
    second = accumulator.push("s1", 1, 1.0, "s2", False)
    third = accumulator.push("s2", 0, 1.0, "s3", False)

    assert first == []
    assert second == []
    assert len(third) == 1

    state, action, reward, next_state, done = third[0]
    assert state == "s0"
    assert action == 0
    assert reward == pytest.approx(1.0 + 0.9 + 0.81)
    assert next_state == "s3"
    assert done is False

    early_done = NStepTransitionAccumulator(n_steps=3, gamma=0.9)
    assert early_done.push("a0", 0, 1.0, "a1", False) == []
    flushed = early_done.push("a1", 1, 1.0, "a2", True)

    assert len(flushed) == 2
    assert flushed[0][0] == "a0"
    assert flushed[0][2] == pytest.approx(1.0 + 0.9)
    assert flushed[0][3] == "a2"
    assert flushed[0][4] is True
    assert flushed[1][0] == "a1"
    assert flushed[1][2] == pytest.approx(1.0)
    assert flushed[1][3] == "a2"
    assert flushed[1][4] is True

    partial = NStepTransitionAccumulator(n_steps=3, gamma=0.9)
    assert partial.push("p0", 0, 1.0, "p1", False) == []
    leftover = partial.flush()
    assert len(leftover) == 1
    assert leftover[0][0] == "p0"
    assert leftover[0][2] == pytest.approx(1.0)
    assert leftover[0][4] is False


def test_rainbow_agent_select_action_and_learn_step_metrics_are_finite() -> None:
    agent = RainbowDQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        device="cpu",
    )
    obs = np.random.randn(OBS_DIM).astype(np.float32)

    action = agent.select_action(obs)
    assert action in {0, 1}

    _fill_agent_buffer(agent, count=32)
    metrics = agent.learn_step()

    for key in ("loss", "q_mean", "td_error_mean", "beta"):
        assert key in metrics
        assert math.isfinite(metrics[key])


def test_c51_projection_conserves_probability_mass() -> None:
    agent = RainbowDQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        device="cpu",
        n_atoms=N_ATOMS,
        v_min=-5.0,
        v_max=5.0,
    )

    next_dist = torch.full((BATCH_SIZE, N_ATOMS), 1.0 / N_ATOMS, dtype=torch.float32)
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    projected = agent._project_distribution(next_dist, rewards, dones)

    assert projected.shape == (BATCH_SIZE, N_ATOMS)
    assert torch.allclose(projected.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)


def test_rainbow_agent_save_load_preserves_training_state(tmp_path) -> None:
    agent = RainbowDQNAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        device="cpu",
    )
    _fill_agent_buffer(agent, count=32)
    metrics = agent.learn_step()
    assert metrics

    checkpoint = agent.save(tmp_path / "rainbow.pt")
    loaded = RainbowDQNAgent.load(
        checkpoint,
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        device="cpu",
    )

    assert loaded.steps == agent.steps
    assert len(loaded.buffer) == len(agent.buffer)
    assert loaded.buffer.beta == pytest.approx(agent.buffer.beta)
    for p_loaded, p_agent in zip(loaded.q_online.parameters(), agent.q_online.parameters()):
        assert torch.allclose(p_loaded, p_agent)


def test_train_rainbow_dqn_smoke(tmp_path) -> None:
    config = _rainbow_config()
    result = train_rainbow_dqn(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], RainbowDQNAgent)
    assert len(result["history"]["episode_rewards"]) == 3
    assert "step_q_means" in result["history"]
    assert "step_td_error_means" in result["history"]
    assert "step_betas" in result["history"]
