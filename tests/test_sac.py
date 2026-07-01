"""Tests pour le module SAC (Soft Actor-Critic)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

import rl_from_scratch  # noqa: F401 - déclenche l'auto-découverte des registres
from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY, load_config
from rl_from_scratch.sac.agent import SACAgent
from rl_from_scratch.sac.buffer import ContinuousReplayBuffer
from rl_from_scratch.sac.config import SACConfig
from rl_from_scratch.sac.network import SquashedGaussianActor
from rl_from_scratch.sac.training import evaluate, train_sac


OBS_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 32
BATCH_SIZE = 4
BUFFER_CAPACITY = 100

ACTION_LOW = np.array([-2.0, -0.5], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 3.0], dtype=np.float32)


def _make_sac_agent(**kwargs) -> SACAgent:
    """Crée un SACAgent minimal pour les tests."""
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        alpha=0.2,
        auto_tune_alpha=True,
        alpha_lr=3e-4,
        target_entropy=None,
        log_std_min=-5.0,
        log_std_max=1.0,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        device="cpu",
    )
    defaults.update(kwargs)
    return SACAgent(**defaults)


def _fill_buffer(agent: SACAgent, n: int = BATCH_SIZE + 1) -> None:
    """Remplit le replay buffer avec des transitions synthétiques."""
    rng = np.random.default_rng(0)
    for _ in range(n):
        obs = rng.normal(size=(OBS_DIM,)).astype(np.float32)
        action = rng.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)
        next_obs = rng.normal(size=(OBS_DIM,)).astype(np.float32)
        reward = float(rng.normal())
        done = bool(rng.integers(0, 2))
        agent.store_transition(obs, action, reward, next_obs, done)


def test_squashed_gaussian_actor_sample_shapes_and_finite_log_prob() -> None:
    """sample retourne actions, log-probs et actions déterministes aux bonnes formes."""
    actor = SquashedGaussianActor(
        OBS_DIM,
        ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        log_std_min=-5.0,
        log_std_max=1.0,
    )
    obs = torch.randn(16, OBS_DIM)

    actions, log_probs, deterministic_actions = actor.sample(obs)

    assert actions.shape == (16, ACTION_DIM)
    assert log_probs.shape == (16,)
    assert deterministic_actions.shape == (16, ACTION_DIM)
    assert torch.isfinite(log_probs).all()


def test_squashed_gaussian_actor_respects_vector_bounds() -> None:
    """L'acteur respecte des bornes différentes pour chaque dimension d'action."""
    actor = SquashedGaussianActor(
        OBS_DIM,
        ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
    )

    with torch.no_grad():
        actions, _, deterministic_actions = actor.sample(torch.randn(64, OBS_DIM))

    low = torch.as_tensor(ACTION_LOW)
    high = torch.as_tensor(ACTION_HIGH)
    assert (actions >= low - 1e-5).all()
    assert (actions <= high + 1e-5).all()
    assert (deterministic_actions >= low - 1e-5).all()
    assert (deterministic_actions <= high + 1e-5).all()


def test_squashed_gaussian_actor_uses_reparameterized_samples() -> None:
    """Le sample SAC garde un chemin de gradient vers les paramètres de l'acteur."""
    actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM, hidden_dim=HIDDEN_DIM)
    obs = torch.randn(8, OBS_DIM)

    actions, log_probs, _ = actor.sample(obs)
    loss = (actions.pow(2).mean() + log_probs.mean())
    loss.backward()

    grads = [p.grad for p in actor.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_squashed_gaussian_actor_forward_clamps_log_std() -> None:
    """forward expose un log_std borné, indispensable pour éviter std=0/inf."""
    actor = SquashedGaussianActor(
        OBS_DIM,
        ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        log_std_min=-3.0,
        log_std_max=-1.0,
    )

    _, log_std = actor(torch.randn(32, OBS_DIM))

    assert (log_std >= -3.0).all()
    assert (log_std <= -1.0).all()


def test_sac_agent_deterministic_action_is_stable() -> None:
    """Le mode déterministe d'évaluation doit être reproductible."""
    agent = _make_sac_agent()
    obs = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)

    action_a = agent.select_action(obs, deterministic=True)
    action_b = agent.select_action(obs, deterministic=True)

    np.testing.assert_allclose(action_a, action_b)


def test_sac_agent_select_action_respects_vector_bounds() -> None:
    """select_action respecte les bornes vectorielles en mode stochastic et eval."""
    agent = _make_sac_agent()
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    stochastic = agent.select_action(obs, deterministic=False)
    deterministic = agent.select_action(obs, deterministic=True)

    assert stochastic.shape == (ACTION_DIM,)
    assert deterministic.shape == (ACTION_DIM,)
    assert np.all(stochastic >= ACTION_LOW - 1e-5)
    assert np.all(stochastic <= ACTION_HIGH + 1e-5)
    assert np.all(deterministic >= ACTION_LOW - 1e-5)
    assert np.all(deterministic <= ACTION_HIGH + 1e-5)


def test_sac_agent_has_no_actor_target_and_uses_continuous_replay_buffer() -> None:
    """SAC ne doit pas avoir d'actor target, contrairement à DDPG/TD3."""
    agent = _make_sac_agent()

    assert not hasattr(agent, "actor_target")
    assert isinstance(agent.replay_buffer, ContinuousReplayBuffer)


def test_sac_agent_store_transition_and_learn_step_keys() -> None:
    """learn_step retourne les métriques SAC attendues quand le buffer est suffisant."""
    agent = _make_sac_agent()
    _fill_buffer(agent)

    metrics = agent.learn_step()

    expected = {
        "critic_loss",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "entropy",
        "q1_mean",
        "q2_mean",
        "q_gap",
        "target_q_mean",
        "log_prob_mean",
    }
    assert expected.issubset(metrics)
    assert all(np.isfinite(float(metrics[key])) for key in expected)


def test_sac_agent_learn_step_empty_when_buffer_too_small() -> None:
    """Avant batch_size transitions, learn_step ne doit pas faire d'update partielle."""
    agent = _make_sac_agent(batch_size=BATCH_SIZE)
    _fill_buffer(agent, n=BATCH_SIZE - 1)

    assert agent.learn_step() == {}


def test_sac_replay_buffer_samples_float_actions_with_expected_shapes() -> None:
    """Le replay buffer partagé reste compatible avec les actions continues SAC."""
    agent = _make_sac_agent()
    _fill_buffer(agent, n=BATCH_SIZE + 2)

    states, actions, rewards, next_states, dones = agent.replay_buffer.sample(BATCH_SIZE)

    assert states.shape == (BATCH_SIZE, OBS_DIM)
    assert actions.shape == (BATCH_SIZE, ACTION_DIM)
    assert rewards.shape == (BATCH_SIZE,)
    assert next_states.shape == (BATCH_SIZE, OBS_DIM)
    assert dones.shape == (BATCH_SIZE,)
    assert actions.dtype == torch.float32


def test_sac_agent_auto_tune_alpha_changes_alpha() -> None:
    """Avec auto_tune_alpha=True, une update doit ajuster alpha."""
    agent = _make_sac_agent(auto_tune_alpha=True)
    _fill_buffer(agent, n=BATCH_SIZE * 4)
    before = agent.alpha

    agent.learn_step()

    assert agent.alpha != pytest.approx(before)


def test_sac_agent_fixed_alpha_stays_constant() -> None:
    """Avec auto_tune_alpha=False, alpha reste fixe et alpha_loss vaut 0."""
    agent = _make_sac_agent(auto_tune_alpha=False, alpha=0.37)
    _fill_buffer(agent, n=BATCH_SIZE * 4)

    metrics = agent.learn_step()

    assert agent.alpha == pytest.approx(0.37)
    assert metrics["alpha"] == pytest.approx(0.37)
    assert metrics["alpha_loss"] == pytest.approx(0.0)


def test_sac_agent_save_load_roundtrip_preserves_metadata(tmp_path) -> None:
    """save/load restaure poids, alpha, log_std bounds et bornes vectorielles."""
    agent = _make_sac_agent(
        alpha=0.31,
        log_std_min=-7.0,
        log_std_max=0.5,
        auto_tune_alpha=True,
    )
    _fill_buffer(agent, n=BATCH_SIZE * 4)
    agent.learn_step()

    path = tmp_path / "sac.pt"
    agent.save(path)
    loaded = SACAgent.load(path, device="cpu")

    assert loaded.alpha == pytest.approx(agent.alpha)
    assert loaded.actor.log_std_min == pytest.approx(-7.0)
    assert loaded.actor.log_std_max == pytest.approx(0.5)
    np.testing.assert_allclose(loaded._action_low_np, ACTION_LOW)
    np.testing.assert_allclose(loaded._action_high_np, ACTION_HIGH)
    assert not hasattr(loaded, "actor_target")


def test_sac_agent_fixed_alpha_save_load_roundtrip(tmp_path) -> None:
    """Le checkpoint fixe alpha restaure un SAC sans log_alpha apprenable."""
    agent = _make_sac_agent(auto_tune_alpha=False, alpha=0.41)
    path = tmp_path / "sac_fixed_alpha.pt"

    agent.save(path)
    loaded = SACAgent.load(path, device="cpu")

    assert loaded.auto_tune_alpha is False
    assert loaded.alpha == pytest.approx(0.41)
    assert not hasattr(loaded, "log_alpha")


def test_sac_evaluate_uses_deterministic_policy_and_reports_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate doit agréger les métriques publiques sur un env séparé."""

    class _EvalEnv:
        def __init__(self) -> None:
            import gymnasium as gym
            self.action_space = gym.spaces.Box(
                low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32,
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
            )
            self.reset_calls: list[int | None] = []
            self.step_calls = 0

        def reset(self, seed=None):
            self.reset_calls.append(seed)
            self.step_calls = 0
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            self.step_calls += 1
            return (
                np.ones(OBS_DIM, dtype=np.float32),
                1.5,
                True,
                False,
                {},
            )

        def close(self) -> None:
            pass

    eval_env = _EvalEnv()
    monkeypatch.setattr(
        "rl_from_scratch.sac.training.make_env",
        lambda env_id, seed, render: eval_env,
    )

    agent = _make_sac_agent()
    calls: list[bool] = []

    def _select_action(obs, deterministic=False):
        calls.append(deterministic)
        return np.array([0.2, -0.1], dtype=np.float32)

    monkeypatch.setattr(agent, "select_action", _select_action)

    metrics = evaluate(
        agent=agent,
        env_id="Pendulum-v1",
        n_episodes=3,
        seed=7,
        solved_reward=1.0,
        max_steps=5,
    )

    assert calls == [True, True, True]
    assert eval_env.reset_calls == [7, 8, 9]
    assert metrics["mean_reward"] == pytest.approx(1.5)
    assert metrics["std_reward"] == pytest.approx(0.0)
    assert metrics["min_reward"] == pytest.approx(1.5)
    assert metrics["max_reward"] == pytest.approx(1.5)
    assert metrics["mean_length"] == pytest.approx(1.0)
    assert metrics["success_rate"] == pytest.approx(1.0)


def test_sac_config_and_agent_are_registered() -> None:
    """L'auto-discovery expose SAC au loader YAML et au CLI."""
    assert CONFIG_REGISTRY["sac"] is SACConfig
    assert AGENT_FACTORIES["sac"] is train_sac


def test_sac_config_validation_rejects_invalid_values() -> None:
    """La config refuse les hyperparamètres invalides importants."""
    with pytest.raises(ValueError, match="alpha"):
        SACConfig(alpha=0.0)
    with pytest.raises(ValueError, match="log_std_min"):
        SACConfig(log_std_min=1.0, log_std_max=1.0)


def test_sac_yaml_loads_as_sac_config() -> None:
    """Le YAML HalfCheetah officiel charge en SACConfig."""
    config = load_config("configs/sac/sac_halfcheetah.yaml")

    assert isinstance(config, SACConfig)
    assert config.approach == "sac"
    assert config.env_id == "HalfCheetah-v5"


def test_sac_training_smoke(tmp_path) -> None:
    """train_sac avec un run court Pendulum termine et produit l'historique SAC."""
    config = SACConfig(
        env_id="Pendulum-v1",
        total_timesteps=500,
        hidden_dim=32,
        batch_size=32,
        buffer_capacity=1000,
        start_steps=50,
        update_after=50,
        checkpoint_every=500,
        device="cpu",
    )
    result = train_sac(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    history = result["history"]
    assert isinstance(history, dict)
    assert "step_alphas" in history
    assert "step_entropies" in history
    assert "step_alpha_losses" in history
    assert "step_log_prob_means" in history
    assert "step_q1_means" in history
    assert "step_q2_means" in history
    assert len(history["episode_rewards"]) > 0


def test_sac_package_does_not_import_sibling_algorithms() -> None:
    """Le package SAC ne doit dépendre d'aucun sous-package d'algorithme frère."""
    sac_dir = Path(__file__).resolve().parents[1] / "src" / "rl_from_scratch" / "sac"

    for path in sac_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module is not None else []
            else:
                continue

            assert all(
                name is None
                or not name.startswith("rl_from_scratch.deterministic_actor_critic")
                for name in names
            ), f"SAC sibling import found in {path.name}: {names}"
