"""Tests for the deep_q module (DQN and Double DQN)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_from_scratch.deep_q.agent import DQNAgent, DoubleDQNAgent
from rl_from_scratch.deep_q.buffer import ReplayBuffer
from rl_from_scratch.deep_q.config import DQNConfig, DoubleDQNConfig
from rl_from_scratch.deep_q.network import QNetwork
from rl_from_scratch.deep_q.training import train_dqn, train_double_dqn


# ------------------------------------------------------------------
# QNetwork
# ------------------------------------------------------------------


def test_qnetwork_output_shape() -> None:
    """QNetwork(4, 2) with a batch of 8 produces shape (8, 2)."""
    net = QNetwork(obs_dim=4, n_actions=2)
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


# ------------------------------------------------------------------
# ReplayBuffer
# ------------------------------------------------------------------


def test_replay_buffer_sample_shapes() -> None:
    """Push 100 transitions, sample 8, verify tensor shapes and dtypes."""
    buf = ReplayBuffer(capacity=200)
    obs_dim = 4
    for _ in range(100):
        s = np.random.randn(obs_dim).astype(np.float32)
        a = int(np.random.randint(2))
        r = float(np.random.randn())
        ns = np.random.randn(obs_dim).astype(np.float32)
        d = bool(np.random.rand() > 0.5)
        buf.push(s, a, r, ns, d)

    assert len(buf) == 100

    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, obs_dim)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, obs_dim)
    assert dones.shape == (8,)

    assert states.dtype == torch.float32
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.float32


# ------------------------------------------------------------------
# DQNAgent
# ------------------------------------------------------------------


def test_dqn_agent_select_action_in_bounds() -> None:
    """Actions returned by the agent are in {0, 1}."""
    agent = DQNAgent(obs_dim=4, n_actions=2, epsilon=0.5, device="cpu")
    obs = np.random.randn(4).astype(np.float32)

    for _ in range(20):
        action_stochastic = agent.select_action(obs)
        assert action_stochastic in {0, 1}

    action_det = agent.select_action(obs, deterministic=True)
    assert action_det in {0, 1}


def test_dqn_agent_learn_step_returns_loss() -> None:
    """After filling the buffer, learn_step returns {'loss': float}."""
    agent = DQNAgent(obs_dim=4, n_actions=2, batch_size=16, device="cpu")
    obs_dim = 4

    # Fill buffer with enough transitions
    for _ in range(100):
        s = np.random.randn(obs_dim).astype(np.float32)
        a = int(np.random.randint(2))
        r = float(np.random.randn())
        ns = np.random.randn(obs_dim).astype(np.float32)
        d = bool(np.random.rand() > 0.8)
        agent.store_transition(s, a, r, ns, d)

    result = agent.learn_step()
    assert "loss" in result
    assert isinstance(result["loss"], float)


# ------------------------------------------------------------------
# DoubleDQNAgent — correctness of action selection vs evaluation
# ------------------------------------------------------------------


def test_double_dqn_uses_online_for_selection() -> None:
    """Verify that Double DQN selects actions with q_online but evaluates with q_target.

    Set up q_online to prefer action 0 and q_target to prefer action 1 for
    next_states.  In vanilla DQN, the target would use q_target's max
    (action 1).  In Double DQN, the target should use q_target's value for
    action 0 (online's argmax), which differs.
    """
    obs_dim = 4
    n_actions = 2

    agent = DoubleDQNAgent(device="cpu", 
        obs_dim=obs_dim, n_actions=n_actions, gamma=0.99, batch_size=1
    )

    # Manually set weights so q_online and q_target disagree
    with torch.no_grad():
        # Make q_online output [10, 0] for any input -> prefers action 0
        for param in agent.q_online.parameters():
            param.zero_()
        # Set last layer bias: action 0 = 10, action 1 = 0
        agent.q_online.net[-1].bias.copy_(torch.tensor([10.0, 0.0]))

        # Make q_target output [5, 20] for any input -> prefers action 1
        for param in agent.q_target.parameters():
            param.zero_()
        agent.q_target.net[-1].bias.copy_(torch.tensor([5.0, 20.0]))

    # Create a single non-terminal transition
    state = np.zeros(obs_dim, dtype=np.float32)
    next_state = np.zeros(obs_dim, dtype=np.float32)
    reward = 1.0
    action = 0
    done = False

    agent.store_transition(state, action, reward, next_state, done)

    # Manually compute what the Double DQN target should be:
    # online argmax for next_state -> action 0 (since online outputs [10, 0])
    # target value at action 0 -> 5.0
    # target = reward + gamma * 5.0 = 1.0 + 0.99 * 5.0 = 5.95
    expected_target = reward + 0.99 * 5.0

    # For vanilla DQN the target would be:
    # target max of q_target -> action 1 value = 20.0
    # target = reward + gamma * 20.0 = 1.0 + 0.99 * 20.0 = 20.8
    vanilla_target = reward + 0.99 * 20.0

    # Compute the actual loss from the Double DQN agent
    batch = agent.buffer.sample(1)
    loss = agent._compute_loss(batch)

    # q_values for state, action 0 via online = 10.0
    # Double DQN target = 5.95
    # MSE loss = (10.0 - 5.95)^2 = 16.4025
    expected_loss = (10.0 - expected_target) ** 2

    # Vanilla DQN target = 20.8
    # MSE loss = (10.0 - 20.8)^2 = 116.64
    vanilla_loss = (10.0 - vanilla_target) ** 2

    assert abs(loss.item() - expected_loss) < 0.01, (
        f"Double DQN loss {loss.item():.4f} != expected {expected_loss:.4f}. "
        f"Got vanilla DQN loss {vanilla_loss:.4f} instead?"
    )
    assert abs(loss.item() - vanilla_loss) > 1.0, (
        "Loss matches vanilla DQN — Double DQN is not using online for selection."
    )


# ------------------------------------------------------------------
# Training smoke tests
# ------------------------------------------------------------------


def test_dqn_training_smoke(tmp_path) -> None:
    """train_dqn with 3 episodes completes and returns expected keys."""
    config = DQNConfig(episodes=3, checkpoint_every=3)
    result = train_dqn(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], DQNAgent)
    assert len(result["history"]["episode_rewards"]) == 3


def test_double_dqn_training_smoke(tmp_path) -> None:
    """train_double_dqn with 3 episodes completes and returns expected keys."""
    config = DoubleDQNConfig(episodes=3, checkpoint_every=3)
    result = train_double_dqn(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], DoubleDQNAgent)
    assert len(result["history"]["episode_rewards"]) == 3
