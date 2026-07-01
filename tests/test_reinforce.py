"""Tests pour le module reinforce (REINFORCE et REINFORCE avec baseline)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_from_scratch.reinforce.agent import ReinforceAgent, ReinforceBaselineAgent
from rl_from_scratch.reinforce.config import ReinforceBaselineConfig, ReinforceConfig
from rl_from_scratch.reinforce.network import PolicyNetwork, ValueNetwork
from rl_from_scratch.reinforce.training import train_reinforce, train_reinforce_baseline


# ------------------------------------------------------------------
# PolicyNetwork
# ------------------------------------------------------------------


def test_policy_network_output_shape() -> None:
    """PolicyNetwork(4, 2) avec un batch de 8 produit la forme (8, 2)."""
    net = PolicyNetwork(obs_dim=4, n_actions=2)
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


# ------------------------------------------------------------------
# ValueNetwork
# ------------------------------------------------------------------


def test_value_network_output_shape() -> None:
    """ValueNetwork(4) avec un batch de 8 produit la forme (8,)."""
    net = ValueNetwork(obs_dim=4)
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8,)


# ------------------------------------------------------------------
# ReinforceAgent
# ------------------------------------------------------------------


def test_reinforce_agent_select_action_in_bounds() -> None:
    """Les actions retournées par l'agent sont dans {0, 1}."""
    agent = ReinforceAgent(obs_dim=4, n_actions=2, device="cpu")
    obs = np.random.randn(4).astype(np.float32)

    for _ in range(20):
        action = agent.select_action(obs)
        assert action in {0, 1}

    action_det = agent.select_action(obs, deterministic=True)
    assert action_det in {0, 1}


def test_reinforce_agent_select_action_stochastic() -> None:
    """Sur de nombreux échantillons, l'agent explore les deux actions."""
    agent = ReinforceAgent(obs_dim=4, n_actions=2, device="cpu")
    obs = np.random.randn(4).astype(np.float32)

    actions = {agent.select_action(obs) for _ in range(100)}
    assert 0 in actions, "L'action 0 n'a jamais été sélectionnée."
    assert 1 in actions, "L'action 1 n'a jamais été sélectionnée."


def test_reinforce_agent_learn_step_returns_loss() -> None:
    """Après un épisode complet, learn_step retourne {'policy_loss': float}."""
    agent = ReinforceAgent(obs_dim=4, n_actions=2, device="cpu")
    obs = np.random.randn(4).astype(np.float32)

    # Simule un épisode de 10 pas
    for _ in range(10):
        action = agent.select_action(obs)
        agent.store_transition(obs, action, 1.0, obs, False)

    result = agent.learn_step()
    assert "policy_loss" in result
    assert isinstance(result["policy_loss"], float)


def test_reinforce_policy_loss_uses_elementwise_time_products() -> None:
    """La loss doit multiplier log_prob_t par return_t sans broadcasting (T, T)."""
    agent = ReinforceAgent(obs_dim=4, n_actions=2, device="cpu")
    agent.log_probs = [
        torch.tensor([-0.7], requires_grad=True),
        torch.tensor([-0.8], requires_grad=True),
        torch.tensor([-0.9], requires_grad=True),
    ]
    returns = torch.tensor([1.0, 0.0, -1.0])

    loss = agent._compute_policy_loss(returns)
    expected = -(torch.tensor([-0.7, -0.8, -0.9]) * returns).mean()

    assert torch.allclose(loss.detach(), expected)
    assert not torch.isclose(loss.detach(), torch.tensor(0.0))


# ------------------------------------------------------------------
# ReinforceBaselineAgent
# ------------------------------------------------------------------


def test_reinforce_baseline_subtracts_value() -> None:
    """L'agent baseline retourne {'policy_loss': float, 'value_loss': float}."""
    agent = ReinforceBaselineAgent(obs_dim=4, n_actions=2, device="cpu")
    obs = np.random.randn(4).astype(np.float32)

    # Simule un épisode de 10 pas
    for _ in range(10):
        action = agent.select_action(obs)
        agent.store_transition(obs, action, 1.0, obs, False)

    result = agent.learn_step()
    assert "policy_loss" in result
    assert "value_loss" in result
    assert isinstance(result["policy_loss"], float)
    assert isinstance(result["value_loss"], float)


# ------------------------------------------------------------------
# Smoke tests d'entraînement
# ------------------------------------------------------------------


def test_reinforce_training_smoke(tmp_path) -> None:
    """train_reinforce avec 3 épisodes termine et retourne les clés attendues."""
    config = ReinforceConfig(episodes=3, checkpoint_every=3)
    result = train_reinforce(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], ReinforceAgent)
    assert len(result["history"]["episode_rewards"]) == 3


def test_reinforce_baseline_training_smoke(tmp_path) -> None:
    """train_reinforce_baseline avec 3 épisodes termine et retourne les clés attendues."""
    config = ReinforceBaselineConfig(episodes=3, checkpoint_every=3)
    result = train_reinforce_baseline(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], ReinforceBaselineAgent)
    assert len(result["history"]["episode_rewards"]) == 3
