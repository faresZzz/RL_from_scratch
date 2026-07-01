"""Tests for the autonomous dyna package."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY
from rl_from_scratch.dyna.agent import DeepDynaAgent, DynaQAgent, DynaQPlusAgent
from rl_from_scratch.dyna.buffer import ReplayBuffer
from rl_from_scratch.dyna.config import DeepDynaConfig, DynaQConfig, DynaQPlusConfig
from rl_from_scratch.dyna.model import TabularWorldModel
from rl_from_scratch.dyna.network import NeuralDynamicsModel, QNetwork
import gymnasium
from rl_from_scratch.dyna.reporting import DynaReporting
from rl_from_scratch.dyna.training import (
    CartPoleDiscretizer,
    train_deep_dyna,
    train_dyna_q,
    train_dyna_q_plus,
)


def _disable_visual_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable figure generation for short tests.

    record_greedy_episode is patched globally via conftest autouse fixture.
    """
    import rl_from_scratch.dyna.reporting as _dyna_reporting

    monkeypatch.setattr(
        _dyna_reporting,
        "generate_training_figures",
        lambda *args, **kwargs: [],
    )


def _fill_deep_buffer(agent: DeepDynaAgent, n: int = 8) -> None:
    for index in range(n):
        obs = np.array([index, index + 1, index + 2, index + 3], dtype=np.float32)
        next_obs = obs + np.array([0.5, -0.25, 0.75, 0.0], dtype=np.float32)
        action = index % agent.n_actions
        reward = float(index % 3)
        done = bool(index % 5 == 0)
        agent.store_transition(obs, action, reward, next_obs, done)


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before: list[torch.Tensor], after_module: torch.nn.Module) -> bool:
    after = list(after_module.parameters())
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def _make_deep_dyna_agent(*, seed: int = 0, target_update_freq: int = 100) -> DeepDynaAgent:
    torch.manual_seed(seed)
    return DeepDynaAgent(
        obs_dim=4,
        n_actions=2,
        hidden_dim=16,
        batch_size=4,
        buffer_capacity=64,
        start_learning_after=4,
        imagined_updates=2,
        model_train_steps=1,
        target_update_freq=target_update_freq,
        device="cpu",
        rng=np.random.default_rng(seed),
    )


def test_replay_buffer_sampling_uses_local_rng_seed() -> None:
    transitions = [
        (
            np.array([index, index + 1], dtype=np.float32),
            index % 2,
            float(index),
            np.array([index + 10, index + 11], dtype=np.float32),
            bool(index % 3 == 0),
        )
        for index in range(10)
    ]
    buffer_a = ReplayBuffer(capacity=16, rng=np.random.default_rng(7))
    buffer_b = ReplayBuffer(capacity=16, rng=np.random.default_rng(7))
    buffer_c = ReplayBuffer(capacity=16, rng=np.random.default_rng(9))

    for transition in transitions:
        buffer_a.push(*transition)
        buffer_b.push(*transition)
        buffer_c.push(*transition)

    sample_a = buffer_a.sample(5)
    sample_b = buffer_b.sample(5)
    sample_c = buffer_c.sample(5)

    for tensor_a, tensor_b in zip(sample_a, sample_b):
        assert torch.equal(tensor_a, tensor_b)

    assert any(not torch.equal(tensor_a, tensor_c) for tensor_a, tensor_c in zip(sample_a, sample_c))


def test_tabular_world_model_store_and_sample() -> None:
    model = TabularWorldModel()
    rng = np.random.default_rng(0)

    model.update((1, 2, 3, 4), 1, 1.5, (2, 3, 4, 5), False)

    assert len(model) == 1
    assert ((1, 2, 3, 4), 1) in model.seen_keys

    sample = model.sample(rng)
    assert sample is not None
    assert sample.state == (1, 2, 3, 4)
    assert sample.action == 1
    assert sample.reward == pytest.approx(1.5)
    assert sample.next_state == (2, 3, 4, 5)
    assert sample.done is False


def test_dyna_q_real_update_changes_q_table() -> None:
    agent = DynaQAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        planning_steps=0,
        rng=np.random.default_rng(0),
    )
    agent.q_table.zero_()

    metrics = agent.learn_real_transition(
        state=(0, 0, 0, 0),
        action=1,
        reward=1.0,
        next_state=(1, 1, 1, 1),
        done=False,
    )

    assert agent.q_table[(0, 0, 0, 0, 1)].item() > 0.0
    assert metrics["real_td_error"] > 0.0


def test_dyna_q_planning_updates_without_new_real_transition() -> None:
    agent = DynaQAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        planning_steps=1,
        rng=np.random.default_rng(0),
    )
    agent.q_table.zero_()
    agent.world_model.update((0, 0, 0, 0), 0, 1.0, (1, 1, 1, 1), False)

    metrics = agent.planning_step()

    assert agent.q_table[(0, 0, 0, 0, 0)].item() > 0.0
    assert metrics["planning_td_error"] > 0.0


def test_dyna_q_plus_bonus_positive_when_tau_positive() -> None:
    agent = DynaQPlusAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        planning_steps=1,
        kappa=0.01,
        rng=np.random.default_rng(0),
    )
    agent.q_table.zero_()
    agent.world_model.update((0, 0, 0, 0), 0, 0.0, (1, 1, 1, 1), False, time_step=0)
    agent.total_updates = 25

    metrics = agent.planning_step()

    assert metrics["exploration_bonus"] > 0.0


def test_dyna_q_plus_bonus_increases_with_tau() -> None:
    agent = DynaQPlusAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        epsilon=0.0,
        planning_steps=1,
        kappa=0.01,
        rng=np.random.default_rng(0),
    )

    low_tau_bonus = agent._planning_bonus(24)
    agent.total_updates = 100
    high_tau_bonus = agent._planning_bonus(24)

    assert high_tau_bonus > low_tau_bonus


def test_dyna_q_plus_adds_untried_actions_for_known_states() -> None:
    agent = DynaQPlusAgent(
        state_shape=(4, 4, 4, 4),
        action_count=3,
        epsilon=0.0,
        planning_steps=1,
        kappa=0.01,
        rng=np.random.default_rng(0),
    )

    state = (1, 1, 1, 1)
    agent.learn_real_transition(
        state=state,
        action=2,
        reward=1.0,
        next_state=(2, 2, 2, 2),
        done=False,
    )

    assert len(agent.world_model) == agent.action_count
    for action in range(agent.action_count):
        assert (state, action) in agent.world_model.seen_keys


def test_deep_dyna_network_and_model_shapes() -> None:
    q_net = QNetwork(obs_dim=4, n_actions=2, hidden_dim=16)
    model = NeuralDynamicsModel(obs_dim=4, n_actions=2, hidden_dim=16)
    obs = torch.randn(8, 4)
    actions = torch.randint(0, 2, (8,))

    q_values = q_net(obs)
    predicted_next, predicted_reward, done_logits = model(obs, actions)

    assert q_values.shape == (8, 2)
    assert predicted_next.shape == (8, 4)
    assert predicted_reward.shape == (8,)
    assert done_logits.shape == (8,)


def test_deep_dyna_metrics_are_finite() -> None:
    agent = _make_deep_dyna_agent(seed=0)
    _fill_deep_buffer(agent)

    metrics = agent.learn_step()

    expected = {
        "real_td_error",
        "planning_td_error",
        "exploration_bonus",
        "q_loss",
        "model_prediction_loss",
        "reward_prediction_loss",
        "done_prediction_loss",
        "imagined_update_count",
        "model_buffer_size",
    }
    assert expected.issubset(metrics)
    for key, value in metrics.items():
        assert math.isfinite(float(value)), f"{key} is not finite: {value}"


def test_deep_dyna_is_reproducible_with_same_seed_and_batch() -> None:
    agent_a = _make_deep_dyna_agent(seed=123)
    agent_b = _make_deep_dyna_agent(seed=123)
    _fill_deep_buffer(agent_a)
    _fill_deep_buffer(agent_b)

    metrics_a = agent_a.learn_step()
    metrics_b = agent_b.learn_step()

    assert metrics_a == pytest.approx(metrics_b)
    for parameter_a, parameter_b in zip(agent_a.q_online.parameters(), agent_b.q_online.parameters()):
        assert torch.allclose(parameter_a, parameter_b)


def test_deep_dyna_training_is_reproducible_with_same_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_visual_artifacts(monkeypatch)
    config = DeepDynaConfig(
        episodes=1,
        eval_every=1,
        eval_episodes=1,
        checkpoint_every=1,
        batch_size=4,
        buffer_capacity=64,
        start_learning_after=4,
        model_train_steps=1,
        imagined_updates=2,
        hidden_dim=16,
        device="cpu",
    )

    result_a = train_deep_dyna(config, output_dir=str(tmp_path), seed=123)
    result_b = train_deep_dyna(config, output_dir=str(tmp_path), seed=123)

    assert result_a["history"]["episode_rewards"] == result_b["history"]["episode_rewards"]
    for parameter_a, parameter_b in zip(
        result_a["agent"].q_online.parameters(),
        result_b["agent"].q_online.parameters(),
    ):
        assert torch.allclose(parameter_a, parameter_b)


def test_deep_dyna_target_network_syncs_at_configured_frequency() -> None:
    agent = _make_deep_dyna_agent(seed=5, target_update_freq=2)
    _fill_deep_buffer(agent)

    agent.learn_step()
    first_gap = [
        torch.max(torch.abs(online - target)).item()
        for online, target in zip(agent.q_online.parameters(), agent.q_target.parameters())
    ]
    assert any(gap > 0.0 for gap in first_gap)

    agent.learn_step()
    for online, target in zip(agent.q_online.parameters(), agent.q_target.parameters()):
        assert torch.allclose(online, target)


def test_imagined_update_changes_q_network_weights() -> None:
    agent = _make_deep_dyna_agent(seed=11)
    _fill_deep_buffer(agent)
    batch = tuple(t.to(agent.device) for t in agent.replay_buffer.sample(agent.batch_size, rng=agent.rng))
    before = _clone_parameters(agent.q_online)

    td_error = agent.learn_q_from_imagined_batch(batch)

    assert td_error >= 0.0
    assert _parameters_changed(before, agent.q_online)


def test_dynamics_model_learns_simple_transition() -> None:
    agent = _make_deep_dyna_agent(seed=17)
    states = torch.tensor([[0.2, -0.1, 0.4, 0.0]], dtype=torch.float32, device=agent.device).repeat(8, 1)
    actions = torch.zeros(8, dtype=torch.long, device=agent.device)
    next_states = states + 0.5
    rewards = torch.full((8,), 2.0, dtype=torch.float32, device=agent.device)
    dones = torch.zeros(8, dtype=torch.float32, device=agent.device)
    batch = (states, actions, rewards, next_states, dones)

    with torch.no_grad():
        next_before, reward_before, _ = agent.dynamics_model(states, actions)
        error_before = torch.mean(torch.abs(next_before - next_states)).item()
        reward_error_before = torch.mean(torch.abs(reward_before - rewards)).item()

    for _ in range(40):
        agent.learn_model_from_real_batch(batch)

    with torch.no_grad():
        next_after, reward_after, _ = agent.dynamics_model(states, actions)
        error_after = torch.mean(torch.abs(next_after - next_states)).item()
        reward_error_after = torch.mean(torch.abs(reward_after - rewards)).item()

    assert error_after < error_before
    assert reward_error_after < reward_error_before


def test_dyna_agents_save_and_load_round_trip(tmp_path: Path) -> None:
    tabular_agent = DynaQAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        planning_steps=1,
        rng=np.random.default_rng(0),
    )
    tabular_agent.learn_real_transition(
        state=(0, 0, 0, 0),
        action=1,
        reward=1.0,
        next_state=(1, 1, 1, 1),
        done=False,
    )
    tabular_path = tmp_path / "dyna_q.pt"
    loaded_tabular = DynaQAgent.load(
        tabular_agent.save(tabular_path),
        state_shape=(4, 4, 4, 4),
        action_count=2,
        planning_steps=1,
        rng=np.random.default_rng(0),
    )
    assert torch.allclose(tabular_agent.q_table, loaded_tabular.q_table)
    assert loaded_tabular.world_model.seen_keys == tabular_agent.world_model.seen_keys

    plus_agent = DynaQPlusAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        planning_steps=1,
        kappa=0.02,
        rng=np.random.default_rng(0),
    )
    plus_agent.learn_real_transition(
        state=(0, 0, 0, 0),
        action=0,
        reward=0.5,
        next_state=(0, 0, 0, 1),
        done=False,
    )
    plus_path = tmp_path / "dyna_q_plus.pt"
    loaded_plus = DynaQPlusAgent.load(
        plus_agent.save(plus_path),
        state_shape=(4, 4, 4, 4),
        action_count=2,
        planning_steps=1,
        rng=np.random.default_rng(0),
    )
    assert torch.allclose(plus_agent.q_table, loaded_plus.q_table)
    assert loaded_plus.kappa == pytest.approx(plus_agent.kappa)

    deep_agent = _make_deep_dyna_agent(seed=23)
    _fill_deep_buffer(deep_agent)
    deep_agent.learn_step()
    deep_path = tmp_path / "deep_dyna.pt"
    loaded_deep = DeepDynaAgent.load(
        deep_agent.save(deep_path),
        obs_dim=4,
        n_actions=2,
        hidden_dim=16,
        batch_size=4,
        buffer_capacity=64,
        start_learning_after=4,
        imagined_updates=2,
        model_train_steps=1,
        device="cpu",
        rng=np.random.default_rng(23),
    )
    for original, restored in zip(deep_agent.q_online.parameters(), loaded_deep.q_online.parameters()):
        assert torch.allclose(original, restored)
    for original, restored in zip(deep_agent.dynamics_model.parameters(), loaded_deep.dynamics_model.parameters()):
        assert torch.allclose(original, restored)
    assert len(loaded_deep.replay_buffer) == len(deep_agent.replay_buffer)


@pytest.mark.parametrize(
    ("config", "trainer", "agent_type"),
    [
        (
            DynaQConfig(episodes=2, eval_every=1, eval_episodes=1, checkpoint_every=2),
            train_dyna_q,
            DynaQAgent,
        ),
        (
            DynaQPlusConfig(episodes=2, eval_every=1, eval_episodes=1, checkpoint_every=2),
            train_dyna_q_plus,
            DynaQPlusAgent,
        ),
        (
            DeepDynaConfig(
                episodes=2,
                eval_every=1,
                eval_episodes=1,
                checkpoint_every=2,
                batch_size=4,
                buffer_capacity=64,
                start_learning_after=4,
                model_train_steps=1,
                imagined_updates=2,
                hidden_dim=16,
                device="cpu",
            ),
            train_deep_dyna,
            DeepDynaAgent,
        ),
    ],
)
def test_dyna_training_smoke_and_return_contract(
    tmp_path, monkeypatch: pytest.MonkeyPatch, config, trainer, agent_type
) -> None:
    _disable_visual_artifacts(monkeypatch)

    result = trainer(config, output_dir=str(tmp_path), seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    assert isinstance(result["agent"], agent_type)
    assert len(result["history"]["episode_rewards"]) == 2
    assert result["metrics"]["mean_reward"] >= 0.0
    assert result["paths"].run_dir.exists()
    assert "step_exploration_bonuss" not in result["history"]
    assert "step_q_losss" not in result["history"]
    assert "step_exploration_bonuses" in result["history"]
    if isinstance(result["agent"], DeepDynaAgent):
        assert "step_q_losses" in result["history"]


def test_dyna_reporting_uses_human_metric_filenames(tmp_path: Path) -> None:
    history = {
        "episode_rewards": [10.0, 12.0],
        "episode_lengths": [10, 12],
        "step_model_prediction_losses": [0.3, 0.2],
        "step_reward_prediction_losses": [0.4, 0.3],
        "step_done_prediction_losses": [0.5, 0.4],
        "step_exploration_bonuses": [0.1, 0.2],
    }

    figures = DynaReporting().generate_figures(
        history,
        DynaQConfig(),
        tmp_path,
    )
    names = {path.name for path in figures}

    assert "model_prediction_loss.png" in names
    assert "reward_prediction_loss.png" in names
    assert "done_prediction_loss.png" in names
    assert "exploration_bonus.png" in names
    assert "model_prediction_losse.png" not in names
    assert "exploration_bonuse.png" not in names


def test_dyna_registry_entries_exist() -> None:
    assert CONFIG_REGISTRY["dyna_q"] is DynaQConfig
    assert CONFIG_REGISTRY["dyna_q_plus"] is DynaQPlusConfig
    assert CONFIG_REGISTRY["deep_dyna"] is DeepDynaConfig
    assert AGENT_FACTORIES["dyna_q"] is train_dyna_q
    assert AGENT_FACTORIES["dyna_q_plus"] is train_dyna_q_plus
    assert AGENT_FACTORIES["deep_dyna"] is train_deep_dyna


def test_cartpole_discretizer_is_deterministic_and_clips_out_of_range() -> None:
    env = gymnasium.make("CartPole-v1")
    config = DynaQConfig()
    discretizer = CartPoleDiscretizer(config, env.observation_space)
    env.close()

    obs = np.array([0.1, -0.3, 0.05, 0.2], dtype=np.float64)

    # (a) deterministic: same obs gives the same tuple twice
    result_a = discretizer.transform(obs)
    result_b = discretizer.transform(obs)
    assert result_a == result_b

    # (b) result is a 4-tuple and every index is within [0, len(edges)-1]
    assert len(result_a) == 4
    for index, edges in zip(result_a, discretizer.bins):
        assert 0 <= index <= len(edges) - 1

    # (c) extreme out-of-range obs (huge velocities) does not raise and stays in bounds
    extreme_obs = np.array([1e9, 1e9, -1e9, -1e9], dtype=np.float64)
    extreme_result = discretizer.transform(extreme_obs)
    assert len(extreme_result) == 4
    for index, edges in zip(extreme_result, discretizer.bins):
        assert 0 <= index <= len(edges) - 1


def test_dyna_q_epsilon_decays_to_floor_and_stays() -> None:
    agent = DynaQAgent(
        state_shape=(4, 4, 4, 4),
        action_count=2,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.05,
        planning_steps=0,
        rng=np.random.default_rng(0),
    )

    for _ in range(2000):
        agent.episode_ended()

    assert agent.epsilon >= agent.min_epsilon
    assert agent.epsilon == pytest.approx(agent.min_epsilon, abs=1e-9)
