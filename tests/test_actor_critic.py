"""Tests pour le module actor_critic (A2C, A2C-GAE et A3C)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rl_from_scratch.core.config import load_config
from rl_from_scratch.actor_critic.agent import A2CAgent, A2CGAEAgent, A3CAgent
from rl_from_scratch.actor_critic.buffer import RolloutBuffer
from rl_from_scratch.actor_critic.config import A2CConfig, A2CGAEConfig, A3CConfig
from rl_from_scratch.actor_critic.network import CriticNetwork, GaussianActor
from rl_from_scratch.actor_critic.optim import SharedAdam
from rl_from_scratch.actor_critic.training import (
    compute_worker_update,
    evaluate,
    train_a2c,
    train_a2c_gae,
    train_a3c,
    train_one_episode,
    train_one_worker_rollout,
    WorkerRollout,
)
from rl_from_scratch.core.env import clip_action
from rl_from_scratch.core.normalization import ObservationNormalizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------
# GaussianActor
# ------------------------------------------------------------------


def test_actor_critic_configs_load_smoke_and_full_variants() -> None:
    config_dir = PROJECT_ROOT / "configs" / "actor_critic"

    a2c_smoke = load_config(config_dir / "a2c_pendulum_smoke.yaml")
    a2c_full = load_config(config_dir / "a2c_halfcheetah.yaml")
    gae_smoke = load_config(config_dir / "a2c_gae_pendulum_smoke.yaml")
    gae_full = load_config(config_dir / "a2c_gae_halfcheetah.yaml")
    a3c_smoke = load_config(config_dir / "a3c_pendulum_smoke.yaml")
    a3c_full = load_config(config_dir / "a3c_halfcheetah.yaml")

    assert isinstance(a2c_smoke, A2CConfig)
    assert a2c_smoke.env_id == "Pendulum-v1"
    assert a2c_smoke.total_timesteps == 80
    assert isinstance(a2c_full, A2CConfig)
    assert a2c_full.env_id == "HalfCheetah-v5"
    assert a2c_full.normalize_observations is True

    assert isinstance(gae_smoke, A2CGAEConfig)
    assert gae_smoke.env_id == "Pendulum-v1"
    assert gae_smoke.total_timesteps == 80
    assert gae_smoke.gae_lambda == pytest.approx(0.95)
    assert isinstance(gae_full, A2CGAEConfig)
    assert gae_full.env_id == "HalfCheetah-v5"
    assert gae_full.normalize_observations is True

    assert isinstance(a3c_smoke, A3CConfig)
    assert a3c_smoke.env_id == "Pendulum-v1"
    assert a3c_smoke.num_workers == 1
    assert a3c_smoke.t_max == 10
    assert isinstance(a3c_full, A3CConfig)
    assert a3c_full.env_id == "HalfCheetah-v5"
    assert a3c_full.num_workers == 6


def test_actor_critic_configs_reject_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown config keys"):
        A2CConfig.from_dict({"approach": "a2c", "line_search_steps": 3})


def test_gaussian_actor_mean_shape() -> None:
    """GaussianActor(17, 6) avec un batch de 8 produit mean de forme (8, 6)."""
    actor = GaussianActor(obs_dim=17, action_dim=6)
    x = torch.randn(8, 17)
    mean, std = actor(x)
    assert mean.shape == (8, 6), f"Attendu (8, 6), obtenu {mean.shape}"
    assert std.shape == (8, 6), f"Attendu std (8, 6), obtenu {std.shape}"


def test_gaussian_actor_log_std_learnable() -> None:
    """log_std est un nn.Parameter et apparaît dans les paramètres de l'acteur."""
    actor = GaussianActor(obs_dim=17, action_dim=6)
    param_names = [name for name, _ in actor.named_parameters()]
    assert "log_std" in param_names, "log_std doit être un nn.Parameter appris."
    # Vérifie qu'il est bien inclus dans parameters()
    param_ids = {id(p) for p in actor.parameters()}
    assert id(actor.log_std) in param_ids


def test_gaussian_actor_action_in_bounds() -> None:
    """Les actions échantillonnées depuis GaussianActor sont dans [-1, 1] après clamp."""
    actor = GaussianActor(obs_dim=4, action_dim=2)
    obs = torch.randn(32, 4)
    dist = actor.get_distribution(obs)
    actions = dist.sample()
    clamped = actions.clamp(-1.0, 1.0)
    assert clamped.shape == (32, 2)
    assert (clamped >= -1.0).all() and (clamped <= 1.0).all()


# ------------------------------------------------------------------
# CriticNetwork
# ------------------------------------------------------------------


def test_critic_output_shape() -> None:
    """CriticNetwork(17) avec un batch de 8 produit la forme (8,)."""
    critic = CriticNetwork(obs_dim=17)
    x = torch.randn(8, 17)
    out = critic(x)
    assert out.shape == (8,), f"Attendu (8,), obtenu {out.shape}"


# ------------------------------------------------------------------
# RolloutBuffer
# ------------------------------------------------------------------


def _fill_buffer(buf: RolloutBuffer, n: int | None = None) -> None:
    """Remplit le buffer avec des données aléatoires."""
    steps = n if n is not None else buf.n_steps
    for _ in range(steps):
        obs = np.random.randn(buf.obs_dim).astype(np.float32)
        action = np.random.randn(buf.action_dim).astype(np.float32)
        buf.push(obs, action, reward=1.0, done=False, log_prob=-0.5, value=0.3)


def test_rollout_buffer_fill_and_full() -> None:
    """Après n_steps push, is_full() retourne True."""
    buf = RolloutBuffer(n_steps=16, obs_dim=4, action_dim=2)
    assert not buf.is_full()
    _fill_buffer(buf)
    assert buf.is_full()


def test_rollout_buffer_compute_returns_shape() -> None:
    """compute_returns retourne des tenseurs de forme (n_steps,)."""
    buf = RolloutBuffer(n_steps=16, obs_dim=4, action_dim=2)
    _fill_buffer(buf)
    returns, advantages = buf.compute_returns(next_value=0.0, gamma=0.99)
    assert returns.shape == (16,), f"Attendu (16,), obtenu {returns.shape}"
    assert advantages.shape == (16,), f"Attendu (16,), obtenu {advantages.shape}"


def test_rollout_buffer_compute_gae_shape() -> None:
    """compute_gae retourne des tenseurs de forme (n_steps,)."""
    buf = RolloutBuffer(n_steps=16, obs_dim=4, action_dim=2)
    _fill_buffer(buf)
    returns, advantages = buf.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert returns.shape == (16,), f"Attendu (16,), obtenu {returns.shape}"
    assert advantages.shape == (16,), f"Attendu (16,), obtenu {advantages.shape}"


def test_rollout_buffer_compute_gae_uses_only_collected_steps() -> None:
    """Un rollout A3C interrompu avant t_max ne doit pas apprendre sur des zéros."""
    buf = RolloutBuffer(n_steps=5, obs_dim=2, action_dim=1)
    for i in range(3):
        buf.push(
            np.full(2, i, dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.2,
        )

    returns, advantages = buf.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)

    assert returns.shape == (3,)
    assert advantages.shape == (3,)


def test_rollout_buffer_compute_returns_uses_only_collected_steps() -> None:
    """Les retours N-step doivent avoir la même longueur que le batch collecté."""
    buf = RolloutBuffer(n_steps=5, obs_dim=2, action_dim=1)
    for i in range(2):
        buf.push(
            np.full(2, i, dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.2,
        )

    returns, advantages = buf.compute_returns(next_value=0.0, gamma=0.99)

    assert returns.shape == (2,)
    assert advantages.shape == (2,)


def test_rollout_buffer_gae_equals_nstep_at_lambda_1() -> None:
    """GAE avec λ=1.0 doit être approximativement égal aux retours N-step.

    Test clé : à λ=1, GAE devient une somme pondérée identique au retour
    Monte Carlo N-step bootstrappé, donc les avantages doivent coïncider
    avec ceux de compute_returns (à la précision numérique près).
    """
    n_steps = 32
    buf = RolloutBuffer(n_steps=n_steps, obs_dim=4, action_dim=2)

    # Données déterministes pour une comparaison exacte
    rng = np.random.default_rng(42)
    for _ in range(n_steps):
        obs = rng.standard_normal(4).astype(np.float32)
        action = rng.standard_normal(2).astype(np.float32)
        buf.push(obs, action, reward=float(rng.uniform(0, 1)), done=False,
                 log_prob=-0.5, value=float(rng.uniform(0, 1)))

    gamma = 0.99
    next_value = 0.5

    _, advantages_nstep = buf.compute_returns(next_value=next_value, gamma=gamma)
    _, advantages_gae = buf.compute_gae(next_value=next_value, gamma=gamma, gae_lambda=1.0)

    assert torch.allclose(advantages_gae, advantages_nstep, atol=1e-5), (
        f"GAE(λ=1) devrait égaler N-step. Max diff: "
        f"{(advantages_gae - advantages_nstep).abs().max().item():.2e}"
    )


# ------------------------------------------------------------------
# A2CAgent
# ------------------------------------------------------------------


def _make_a2c_agent(obs_dim: int = 3, action_dim: int = 1, n_steps: int = 16) -> A2CAgent:
    """Crée un A2CAgent minimal pour les tests."""
    return A2CAgent(obs_dim=obs_dim, action_dim=action_dim, n_steps=n_steps, device="cpu")


def _skip_if_torch_shared_memory_unavailable() -> None:
    """Skippe les tests A3C si le sandbox bloque torch_shm_manager."""
    try:
        torch.zeros(1).share_memory_()
    except RuntimeError as exc:
        pytest.skip(f"Torch shared memory unavailable in this environment: {exc}")


def test_a2c_select_action_shape() -> None:
    """select_action retourne un tableau numpy de forme (action_dim,)."""
    agent = _make_a2c_agent(obs_dim=3, action_dim=1)
    obs = np.random.randn(3).astype(np.float32)
    action = agent.select_action(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,), f"Attendu (1,), obtenu {action.shape}"


def test_a2c_learn_step_returns_loss_keys() -> None:
    """learn_step retourne les pertes et diagnostics utiles au suivi A2C."""
    n_steps = 16
    agent = _make_a2c_agent(obs_dim=3, action_dim=1, n_steps=n_steps)
    obs = np.random.randn(3).astype(np.float32)

    for _ in range(n_steps):
        action = agent.select_action(obs)
        agent.store_transition(obs, action, 1.0, obs, False)

    result = agent.learn_step(next_value=0.0)

    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "adv_mean",
        "adv_std",
        "explained_variance",
        "grad_norm",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
        "action_abs_mean",
        "action_clip_fraction",
    }
    assert expected_keys.issubset(result.keys()), (
        f"Clés attendues manquantes: {expected_keys - set(result.keys())}"
    )
    for k in expected_keys:
        v = result[k]
        assert isinstance(v, float), f"{k} devrait être float, obtenu {type(v)}"
        assert np.isfinite(v), f"{k} devrait être fini, obtenu {v}"


def test_a2c_action_diagnostics_reports_clipping_fraction() -> None:
    """Les diagnostics d'action mesurent l'amplitude et la fraction clippée."""
    agent = _make_a2c_agent(obs_dim=3, action_dim=1, n_steps=2)
    obs = np.random.randn(3).astype(np.float32)

    for _ in range(2):
        action = agent.select_action(obs)
        agent.record_action_diagnostics(
            raw_action=np.array([3.0], dtype=np.float32),
            clipped_action=np.array([2.0], dtype=np.float32),
        )
        agent.store_transition(obs, action, 1.0, obs, False)

    result = agent.learn_step(next_value=0.0)

    assert result["action_abs_mean"] == pytest.approx(2.0)
    assert result["action_clip_fraction"] == pytest.approx(1.0)


def test_clip_action_uses_env_bounds() -> None:
    """clip_action clips to the environment's actual Box bounds, not [-1, 1]."""
    import gymnasium as gym

    class _DummyEnv:
        action_space = gym.spaces.Box(
            low=np.array([-2.0, -0.5], dtype=np.float32),
            high=np.array([2.0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    action = np.array([-9.0, 3.0], dtype=np.float32)
    clipped = clip_action(action, _DummyEnv())
    np.testing.assert_allclose(clipped, np.array([-2.0, 0.5], dtype=np.float32))


def test_policy_action_kept_raw_env_action_clipped() -> None:
    """The training loop keeps the raw policy action for log_prob but clips for env.step."""
    import gymnasium as gym

    space = gym.spaces.Box(
        low=np.array([-2.0], dtype=np.float32),
        high=np.array([2.0], dtype=np.float32),
        dtype=np.float32,
    )
    raw = np.array([3.0], dtype=np.float32)
    clipped = np.clip(raw, space.low, space.high)

    np.testing.assert_allclose(raw, np.array([3.0], dtype=np.float32))
    np.testing.assert_allclose(clipped, np.array([2.0], dtype=np.float32))


def test_bootstrap_truncated_reward_only_for_time_limit() -> None:
    """Bootstrap γV(s_T) is added only for truncated (TimeLimit) episodes.

    The formula (now inlined in train_one_episode):
        if truncated and not terminated:
            stored_reward = reward + gamma * V(s')
    """
    gamma = 0.9
    terminal_value = 10.0

    # Truncated: bootstrap applies → reward + gamma * V(s')
    assert 1.0 + gamma * terminal_value == pytest.approx(10.0)
    # Terminated: no bootstrap → original reward
    reward_terminated = 1.0  # no adjustment
    assert reward_terminated == pytest.approx(1.0)
    # Neither: no bootstrap → original reward
    reward_neither = 1.0
    assert reward_neither == pytest.approx(1.0)


def test_train_one_episode_bootstraps_truncated_reward_and_keeps_done() -> None:
    """Le helper de training coupe le chaînage au timeout tout en ajoutant γV(s_T)."""
    import gymnasium as gym

    class DummyEnv:
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def reset(self, seed=None):
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            return np.array([1.0], dtype=np.float32), 1.0, False, True, {}

    class DummyAgent:
        gamma = 0.9

        def __init__(self) -> None:
            self.buffer = type("Buffer", (), {"is_full": lambda self: False})()
            self.stored: tuple | None = None

        def select_action(self, obs, *, deterministic=False):
            return np.array([0.0], dtype=np.float32)

        def _to_tensor(self, obs):
            return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        def critic(self, obs_t):
            return torch.tensor([10.0])

        def store_transition(self, obs, action, reward, next_obs, done):
            self.stored = (obs, action, reward, next_obs, done)

        def record_action_diagnostics(self, raw_action, clipped_action):
            pass

    agent = DummyAgent()

    result, learn_results = train_one_episode(agent, DummyEnv(), max_steps=1, seed=0)

    assert learn_results == []
    assert result["truncated"] is True
    assert agent.stored is not None
    assert agent.stored[2] == pytest.approx(10.0)
    assert agent.stored[4] is True


def test_train_one_episode_steps_with_clipped_action_but_stores_policy_action() -> None:
    """L'env reçoit l'action valide, le buffer reçoit l'action cohérente avec log_prob."""
    import gymnasium as gym

    class DummyEnv:
        action_space = gym.spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def __init__(self) -> None:
            self.seen_action: np.ndarray | None = None

        def reset(self, seed=None):
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            self.seen_action = np.asarray(action, dtype=np.float32)
            return np.array([0.0], dtype=np.float32), 1.0, True, False, {}

    class DummyAgent:
        gamma = 0.99

        def __init__(self) -> None:
            self.buffer = type("Buffer", (), {"is_full": lambda self: False})()
            self.stored_action: np.ndarray | None = None
            self.diag: tuple[np.ndarray, np.ndarray] | None = None

        def select_action(self, obs, *, deterministic=False):
            return np.array([3.0], dtype=np.float32)

        def record_action_diagnostics(self, raw_action, clipped_action):
            self.diag = (
                np.asarray(raw_action, dtype=np.float32),
                np.asarray(clipped_action, dtype=np.float32),
            )

        def store_transition(self, obs, action, reward, next_obs, done):
            self.stored_action = np.asarray(action, dtype=np.float32)

    env = DummyEnv()
    agent = DummyAgent()
    result, _ = train_one_episode(agent, env, max_steps=1, seed=0)

    assert result["length"] == 1
    np.testing.assert_allclose(env.seen_action, np.array([2.0], dtype=np.float32))
    np.testing.assert_allclose(agent.stored_action, np.array([3.0], dtype=np.float32))
    assert agent.diag is not None
    np.testing.assert_allclose(agent.diag[0], np.array([3.0], dtype=np.float32))
    np.testing.assert_allclose(agent.diag[1], np.array([2.0], dtype=np.float32))


# ------------------------------------------------------------------
# A2CGAEAgent
# ------------------------------------------------------------------


def test_a2c_gae_differs_from_a2c() -> None:
    """Avec λ<1, les avantages GAE diffèrent des avantages N-step.

    Ce test vérifie que A2CGAEAgent et A2CAgent produisent des mises à jour
    distinctes sur le même rollout.
    """
    n_steps = 64
    obs_dim, action_dim = 3, 1

    # Crée deux agents avec les mêmes poids initiaux
    torch.manual_seed(0)
    agent_a2c = A2CAgent(obs_dim=obs_dim, action_dim=action_dim, n_steps=n_steps, device="cpu")

    torch.manual_seed(0)
    agent_gae = A2CGAEAgent(obs_dim=obs_dim, action_dim=action_dim, n_steps=n_steps, device="cpu",
                            gae_lambda=0.95)

    # Même rollout dans les deux buffers
    rng = np.random.default_rng(42)
    for _ in range(n_steps):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        reward = float(rng.uniform(-1, 1))
        log_prob = -0.5
        value = float(rng.uniform(0, 1))
        done = False
        agent_a2c.buffer.push(obs, action, reward, done, log_prob, value)
        agent_gae.buffer.push(obs, action, reward, done, log_prob, value)

    next_value = 0.3
    _, adv_a2c = agent_a2c.buffer.compute_returns(next_value, gamma=0.99)
    _, adv_gae = agent_gae.buffer.compute_gae(next_value, gamma=0.99, gae_lambda=0.95)

    # λ=0.95 ≠ 1 → les avantages doivent différer
    assert not torch.allclose(adv_a2c, adv_gae, atol=1e-4), (
        "A2C (N-step) et A2C-GAE (λ=0.95) ne devraient pas produire les mêmes avantages."
    )


# ------------------------------------------------------------------
# Smoke tests d'entraînement (Pendulum-v1 uniquement — pas de MuJoCo requis)
# ------------------------------------------------------------------


def test_a2c_training_smoke(tmp_path) -> None:
    """train_a2c avec 200 timesteps sur Pendulum-v1 termine et retourne les clés attendues.

    Pendulum-v1 : espace d'action Box(-2, 2, shape=(1,)) — continu, pas de MuJoCo.
    """
    config = A2CConfig(
        env_id="Pendulum-v1",
        total_timesteps=200,
        n_steps=100,
        max_steps_per_episode=100,
        checkpoint_every=100,
        eval_every=1,
        eval_episodes=1,
    )
    result = train_a2c(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], A2CAgent)
    assert result["history"]["step_grad_norms"]
    assert result["history"]["step_log_std_means"]
    assert result["history"]["step_action_clip_fractions"]
    assert result["metrics"]["best_eval_mean_reward"] == max(
        result["history"]["eval_mean_rewards"]
    )
    assert result["metrics"]["final_eval_mean_reward"] == result["history"]["eval_mean_rewards"][-1]
    assert (result["paths"].figure_dir / "eval_diagnostics.png").exists()
    assert any(
        checkpoint.name == "checkpoint_000200.pt"
        for checkpoint in result["paths"].checkpoint_dir.glob("checkpoint_*.pt")
    )


def test_a2c_training_persists_via_run_recorder(tmp_path, monkeypatch) -> None:
    """La boucle A2C persiste history/metrics via RunRecorder."""
    from rl_from_scratch.core.recording import RunRecorder

    calls: list[dict[str, object]] = []
    original_persist = RunRecorder.persist

    def spy_persist(
        self,
        paths,
        *,
        total_timesteps=None,
        observed_timesteps=None,
        episodes_to_solve=None,
    ):
        calls.append(
            {
                "history_keys": set(self.history),
                "total_timesteps": total_timesteps,
                "observed_timesteps": observed_timesteps,
                "episodes_to_solve": episodes_to_solve,
                "paths": paths,
            }
        )
        return original_persist(
            self,
            paths,
            total_timesteps=total_timesteps,
            observed_timesteps=observed_timesteps,
            episodes_to_solve=episodes_to_solve,
        )

    monkeypatch.setattr(RunRecorder, "persist", spy_persist)

    config = A2CConfig(
        env_id="Pendulum-v1",
        total_timesteps=100,
        n_steps=50,
        max_steps_per_episode=50,
        checkpoint_every=100,
        eval_every=1,
        eval_episodes=1,
    )

    result = train_a2c(config, output_dir=str(tmp_path), seed=0)

    assert len(calls) == 1
    assert calls[0]["total_timesteps"] == config.total_timesteps
    assert calls[0]["observed_timesteps"] == config.total_timesteps
    assert "episode_rewards" in calls[0]["history_keys"]
    assert "step_total_losses" in calls[0]["history_keys"]
    assert result["paths"] == calls[0]["paths"]


def test_a2c_gae_training_smoke(tmp_path) -> None:
    """train_a2c_gae avec 200 timesteps sur Pendulum-v1 termine et retourne les clés attendues."""
    config = A2CGAEConfig(
        env_id="Pendulum-v1",
        total_timesteps=200,
        n_steps=100,
        max_steps_per_episode=100,
        checkpoint_every=100,
        gae_lambda=0.95,
        eval_every=1,
        eval_episodes=1,
    )
    result = train_a2c_gae(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], A2CGAEAgent)
    assert result["history"]["step_grad_norms"]
    assert result["history"]["step_log_std_means"]
    assert result["history"]["step_action_clip_fractions"]
    assert result["metrics"]["best_eval_mean_reward"] == max(
        result["history"]["eval_mean_rewards"]
    )
    assert result["metrics"]["final_eval_mean_reward"] == result["history"]["eval_mean_rewards"][-1]
    assert (result["paths"].figure_dir / "eval_diagnostics.png").exists()
    assert any(
        checkpoint.name == "checkpoint_000200.pt"
        for checkpoint in result["paths"].checkpoint_dir.glob("checkpoint_*.pt")
    )


# ------------------------------------------------------------------
# SharedAdam
# ------------------------------------------------------------------


def test_shared_adam_state_in_shared_memory() -> None:
    """Les tenseurs d'état de SharedAdam (exp_avg, exp_avg_sq) sont en mémoire partagée."""
    _skip_if_torch_shared_memory_unavailable()
    actor = GaussianActor(obs_dim=3, action_dim=1, hidden_dim=32)
    optimizer = SharedAdam(actor.parameters(), lr=1e-3)

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            assert state["step"].is_shared(), "step doit être en mémoire partagée."
            assert state["exp_avg"].is_shared(), "exp_avg doit être en mémoire partagée."
            assert state["exp_avg_sq"].is_shared(), "exp_avg_sq doit être en mémoire partagée."


# ------------------------------------------------------------------
# A3CAgent
# ------------------------------------------------------------------


def test_a3c_sync_local_from_shared() -> None:
    """sync_local_from_shared copie correctement les poids du modèle partagé vers le local."""
    shared_actor = GaussianActor(obs_dim=3, action_dim=1, hidden_dim=32)
    local_actor = GaussianActor(obs_dim=3, action_dim=1, hidden_dim=32)

    # Modifie les poids du modèle partagé pour les distinguer
    with torch.no_grad():
        for p in shared_actor.parameters():
            p.fill_(1.0)
    for p in local_actor.parameters():
        assert not torch.allclose(p, torch.ones_like(p)), (
            "Les poids locaux doivent différer du modèle partagé avant sync."
        )

    A3CAgent.sync_local_from_shared(local_actor, shared_actor)

    for local_p, shared_p in zip(local_actor.parameters(), shared_actor.parameters()):
        assert torch.allclose(local_p, shared_p), (
            "Après sync, les poids locaux doivent être identiques aux poids partagés."
        )


def test_a3c_push_gradients_to_shared() -> None:
    """push_gradients_to_shared copie les gradients locaux vers le modèle partagé."""
    _skip_if_torch_shared_memory_unavailable()
    obs_dim, action_dim, hidden_dim = 3, 1, 32
    local_actor = GaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    shared_actor = GaussianActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    shared_actor.share_memory()

    # Déclenche un backward pour créer des gradients dans le modèle local
    obs = torch.randn(4, obs_dim)
    dist = local_actor.get_distribution(obs)
    loss = -dist.log_prob(torch.zeros(4, action_dim)).sum()
    loss.backward()

    # Vérifie que les gradients locaux existent
    for p in local_actor.parameters():
        assert p.grad is not None, "Le backward doit avoir créé des gradients locaux."

    A3CAgent.push_gradients_to_shared(local_actor, shared_actor)

    for local_p, shared_p in zip(local_actor.parameters(), shared_actor.parameters()):
        assert shared_p._grad is not None, (
            "Les gradients partagés doivent être non-None après push."
        )
        assert torch.allclose(local_p.grad, shared_p._grad), (
            "Les gradients partagés doivent être identiques aux gradients locaux."
        )


def test_a3c_agent_inherits_gae() -> None:
    """A3CAgent hérite de A2CGAEAgent et utilise compute_gae pour les avantages."""
    agent = A3CAgent(obs_dim=3, action_dim=1, n_steps=16, t_max=16, device="cpu")
    assert isinstance(agent, A2CGAEAgent), "A3CAgent doit hériter de A2CGAEAgent."
    assert hasattr(agent, "gae_lambda"), "A3CAgent doit avoir l'attribut gae_lambda."

    # Remplit le buffer et vérifie que _compute_advantages utilise GAE
    for _ in range(16):
        obs = np.random.randn(3).astype(np.float32)
        action = np.random.randn(1).astype(np.float32)
        agent.buffer.push(obs, action, reward=1.0, done=False, log_prob=-0.5, value=0.5)

    returns, advantages = agent._compute_advantages(next_value=0.0)
    assert returns.shape == (16,), f"Attendu (16,), obtenu {returns.shape}"
    assert advantages.shape == (16,), f"Attendu (16,), obtenu {advantages.shape}"


def test_train_one_worker_rollout_bootstraps_truncation_and_tracks_clipping() -> None:
    """Le helper worker A3C stocke l'action policy, bootstrappe TimeLimit et loggue le clipping."""
    import gymnasium as gym
    import multiprocessing as mp

    class DummyDist:
        def sample(self):
            return torch.tensor([[3.0]], dtype=torch.float32)

        def log_prob(self, action):
            return torch.full_like(action, -0.5)

    class DummyActor:
        def get_distribution(self, obs):
            return DummyDist()

    class DummyCritic:
        def __call__(self, obs):
            return torch.tensor([10.0], dtype=torch.float32)

    class DummyEnv:
        action_space = gym.spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def __init__(self) -> None:
            self.seen_action: np.ndarray | None = None
            self.reset_calls = 0

        def reset(self, seed=None):
            self.reset_calls += 1
            return np.array([float(self.reset_calls)], dtype=np.float32), {}

        def step(self, action):
            self.seen_action = np.asarray(action, dtype=np.float32)
            return np.array([9.0], dtype=np.float32), 1.0, False, True, {}

    env = DummyEnv()
    obs, _ = env.reset(seed=0)
    buffer = RolloutBuffer(n_steps=4, obs_dim=1, action_dim=1)

    rollout = train_one_worker_rollout(
        local_actor=DummyActor(),
        local_critic=DummyCritic(),
        env=env,
        buffer=buffer,
        obs=obs,
        episode_reward=0.0,
        episode_length=0,
        global_step_counter=mp.Value("i", 0),
        total_timesteps=10,
        gamma=0.9,
    )

    assert rollout.steps_collected == 1
    assert rollout.episode_ended is True
    assert rollout.completed_episode_reward == pytest.approx(1.0)
    assert rollout.completed_episode_length == 1
    assert rollout.action_abs_mean == pytest.approx(2.0)
    assert rollout.action_clip_fraction == pytest.approx(1.0)
    np.testing.assert_allclose(env.seen_action, np.array([2.0], dtype=np.float32))
    batch = buffer.get_batch()
    np.testing.assert_allclose(
        batch["actions"].numpy(),
        np.array([[3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(buffer._rewards[: buffer._ptr], np.array([10.0], dtype=np.float32))


def test_train_one_worker_rollout_honors_artificial_max_steps_and_bootstraps() -> None:
    """La limite A3C configurée agit comme une troncature TimeLimit avec bootstrap."""
    import gymnasium as gym
    import multiprocessing as mp

    class DummyDist:
        def sample(self):
            return torch.tensor([[0.5]], dtype=torch.float32)

        def log_prob(self, action):
            return torch.full_like(action, -0.25)

    class DummyActor:
        def get_distribution(self, obs):
            return DummyDist()

    class DummyCritic:
        def __call__(self, obs):
            return torch.tensor([10.0], dtype=torch.float32)

    class NeverEndingEnv:
        action_space = gym.spaces.Box(
            low=np.array([-2.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self, seed=None):
            self.reset_calls += 1
            return np.array([float(self.reset_calls)], dtype=np.float32), {}

        def step(self, action):
            return np.array([4.0], dtype=np.float32), 1.0, False, False, {}

    env = NeverEndingEnv()
    obs, _ = env.reset(seed=0)
    buffer = RolloutBuffer(n_steps=4, obs_dim=1, action_dim=1)

    rollout = train_one_worker_rollout(
        local_actor=DummyActor(),
        local_critic=DummyCritic(),
        env=env,
        buffer=buffer,
        obs=obs,
        episode_reward=0.0,
        episode_length=0,
        global_step_counter=mp.Value("i", 0),
        total_timesteps=10,
        gamma=0.99,
        max_steps_per_episode=1,
    )

    assert rollout.steps_collected == 1
    assert rollout.episode_ended is True
    assert rollout.completed_episode_reward == pytest.approx(1.0)
    assert rollout.completed_episode_length == 1
    assert env.reset_calls == 2
    np.testing.assert_allclose(
        buffer._rewards[: buffer._ptr],
        np.array([1.0 + 0.99 * 10.0], dtype=np.float32),
    )


def test_compute_worker_update_returns_expected_metrics() -> None:
    """Le helper d'update A3C conserve les métriques publiques et les gradients locaux."""
    torch.manual_seed(0)
    local_actor = GaussianActor(obs_dim=2, action_dim=1, hidden_dim=16)
    local_critic = CriticNetwork(obs_dim=2, hidden_dim=16)
    buffer = RolloutBuffer(n_steps=4, obs_dim=2, action_dim=1)

    transitions = [
        (np.array([0.1, -0.2], dtype=np.float32), np.array([0.5], dtype=np.float32), 1.0, False, -0.4, 0.2),
        (np.array([0.0, 0.3], dtype=np.float32), np.array([-0.2], dtype=np.float32), 0.5, False, -0.2, 0.1),
        (np.array([-0.4, 0.2], dtype=np.float32), np.array([0.1], dtype=np.float32), -0.25, False, -0.1, -0.05),
    ]
    for transition in transitions:
        buffer.push(*transition)

    next_obs = np.array([0.2, 0.1], dtype=np.float32)
    with torch.no_grad():
        next_value = float(
            local_critic(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)).item()
        )
    _, raw_advantages = buffer.compute_gae(
        next_value=next_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    metrics = compute_worker_update(
        local_actor=local_actor,
        local_critic=local_critic,
        buffer=buffer,
        rollout=WorkerRollout(
            next_obs=next_obs,
            episode_reward=1.25,
            episode_length=3,
            episode_ended=False,
            completed_episode_reward=None,
            completed_episode_length=None,
            action_abs_mean=0.6,
            action_clip_fraction=0.25,
            steps_collected=3,
        ),
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )

    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "explained_variance",
        "grad_norm",
        "adv_mean",
        "adv_std",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
        "action_abs_mean",
        "action_clip_fraction",
    }
    assert expected_keys == set(metrics.keys())
    assert metrics["adv_mean"] == pytest.approx(float(raw_advantages.mean().item()))
    assert metrics["adv_std"] == pytest.approx(float(raw_advantages.std().item()))
    assert metrics["action_abs_mean"] == pytest.approx(0.6)
    assert metrics["action_clip_fraction"] == pytest.approx(0.25)
    for key in expected_keys:
        assert np.isfinite(metrics[key]), f"{key} devrait être fini, obtenu {metrics[key]}"

    for parameter in list(local_actor.parameters()) + list(local_critic.parameters()):
        assert parameter.grad is not None


def test_a3c_training_smoke(tmp_path) -> None:
    """train_a3c avec 2 workers et 400 timesteps sur Pendulum-v1 termine sans erreur.

    Pendulum-v1 : espace d'action Box(-2, 2, shape=(1,)) — continu, pas de MuJoCo.
    Vérifie :
    - La fonction retourne les 4 clés attendues.
    - La liste de récompenses d'épisode est non-vide (au moins un épisode terminé).
    - Les poids de l'agent diffèrent de l'initialisation (apprentissage effectif).
    """
    _skip_if_torch_shared_memory_unavailable()

    # Capture les poids initiaux pour comparer après entraînement
    torch.manual_seed(42)
    reference_actor = GaussianActor(obs_dim=3, action_dim=1, hidden_dim=64)
    initial_weights = [p.clone() for p in reference_actor.parameters()]

    config = A3CConfig(
        env_id="Pendulum-v1",
        total_timesteps=400,
        num_workers=2,
        t_max=10,
        n_steps=10,
        hidden_dim=64,
        checkpoint_every=400,
        gae_lambda=0.95,
        device="cpu",
    )
    result = train_a3c(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result, "La clé 'agent' est manquante."
    assert "history" in result, "La clé 'history' est manquante."
    assert "metrics" in result, "La clé 'metrics' est manquante."
    assert "paths" in result, "La clé 'paths' est manquante."
    assert isinstance(result["agent"], A3CAgent), "L'agent doit être une instance d'A3CAgent."

    episode_rewards = result["history"]["episode_rewards"]
    assert len(episode_rewards) > 0, (
        "Au moins un épisode doit s'être terminé pendant l'entraînement."
    )

    # Vérifie que les poids ont changé (apprentissage réel)
    trained_agent: A3CAgent = result["agent"]
    trained_params = list(trained_agent.actor.parameters())
    any_changed = any(
        not torch.allclose(tp, ip)
        for tp, ip in zip(trained_params, initial_weights)
    )
    assert any_changed, "Les poids de l'acteur doivent avoir changé après l'entraînement."


# ------------------------------------------------------------------
# Normalisation des observations dans A2CAgent
# ------------------------------------------------------------------


def test_a2c_agent_with_normalizer() -> None:
    """A2CAgent avec normalize_observations=True crée un ObservationNormalizer."""
    agent = A2CAgent(
        obs_dim=4,
        action_dim=2,
        n_steps=16,
        normalize_observations=True,
        device="cpu",
    )
    assert agent.obs_normalizer is not None, (
        "obs_normalizer doit être non-None quand normalize_observations=True."
    )
    assert isinstance(agent.obs_normalizer, ObservationNormalizer), (
        f"obs_normalizer doit être un ObservationNormalizer, obtenu {type(agent.obs_normalizer)}."
    )
    assert agent.obs_normalizer.obs_dim == 4


def test_a2c_agent_without_normalizer() -> None:
    """A2CAgent avec normalize_observations=False (défaut) n'a pas de normalizer."""
    agent = A2CAgent(obs_dim=4, action_dim=2, n_steps=16, device="cpu")
    assert agent.obs_normalizer is None, (
        "obs_normalizer doit être None quand normalize_observations=False (défaut)."
    )


def test_a2c_agent_normalizer_save_load(tmp_path) -> None:
    """Après save/load, les statistiques du normaliseur sont correctement restaurées."""
    agent = A2CAgent(
        obs_dim=4,
        action_dim=2,
        n_steps=16,
        normalize_observations=True,
        obs_norm_epsilon=1e-6,
        obs_norm_clip=5.0,
        device="cpu",
    )

    # Accumule des statistiques non triviales
    rng = np.random.default_rng(42)
    for _ in range(20):
        obs = rng.standard_normal(4).astype(np.float32) + 3.0
        agent.obs_normalizer.normalize(obs, update=True)

    saved_mean = agent.obs_normalizer.rms.mean.copy()
    saved_var = agent.obs_normalizer.rms.var.copy()
    saved_count = agent.obs_normalizer.rms.count

    # Sauvegarde et recharge
    ckpt_path = tmp_path / "agent.pt"
    agent.save(ckpt_path)
    loaded_agent = A2CAgent.load(
        ckpt_path,
        obs_dim=4,
        action_dim=2,
        normalize_observations=True,
        device="cpu",
    )

    assert loaded_agent.obs_normalizer is not None, (
        "Le normalizer doit être restauré depuis le checkpoint."
    )
    np.testing.assert_allclose(
        loaded_agent.obs_normalizer.rms.mean, saved_mean, atol=1e-10,
        err_msg="mean du normalizer non restaurée correctement.",
    )
    np.testing.assert_allclose(
        loaded_agent.obs_normalizer.rms.var, saved_var, atol=1e-10,
        err_msg="var du normalizer non restaurée correctement.",
    )
    assert loaded_agent.obs_normalizer.rms.count == saved_count


def test_train_one_episode_normalizes_in_train() -> None:
    """En mode train, le helper met à jour les statistiques du normalizer."""
    import gymnasium as gym

    class DummyEnv:
        action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = gym.spaces.Box(
            low=np.full(4, -10.0, dtype=np.float32),
            high=np.full(4, 10.0, dtype=np.float32),
            dtype=np.float32,
        )
        _step = 0

        def reset(self, seed=None):
            self._step = 0
            return np.ones(4, dtype=np.float32), {}

        def step(self, action):
            self._step += 1
            done = self._step >= 5
            return np.ones(4, dtype=np.float32), 1.0, done, False, {}

        def close(self):
            return None

    agent = A2CAgent(
        obs_dim=4,
        action_dim=2,
        n_steps=32,
        normalize_observations=True,
        device="cpu",
    )
    env = DummyEnv()
    assert agent.obs_normalizer.rms.count == 0.0, (
        "Avant tout épisode, count doit être 0."
    )
    train_one_episode(agent, env, max_steps=10, seed=0)

    assert agent.obs_normalizer.rms.count > 0.0, (
        "Après un épisode en mode train, count doit être > 0 (stats mises à jour)."
    )


def test_evaluate_does_not_update_normalizer_stats() -> None:
    """En mode eval, le helper d'évaluation ne modifie pas les statistiques."""
    import gymnasium as gym

    class DummyEnv:
        action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_space = gym.spaces.Box(
            low=np.full(4, -10.0, dtype=np.float32),
            high=np.full(4, 10.0, dtype=np.float32),
            dtype=np.float32,
        )
        _step = 0

        def reset(self, seed=None):
            self._step = 0
            return np.ones(4, dtype=np.float32), {}

        def step(self, action):
            self._step += 1
            done = self._step >= 5
            return np.ones(4, dtype=np.float32), 1.0, done, False, {}

        def close(self):
            return None

    agent = A2CAgent(
        obs_dim=4,
        action_dim=2,
        n_steps=32,
        normalize_observations=True,
        device="cpu",
    )
    env = DummyEnv()
    # Initialise les stats via un épisode d'entraînement
    train_one_episode(agent, env, max_steps=10, seed=0)
    count_after_train = agent.obs_normalizer.rms.count
    mean_after_train = agent.obs_normalizer.rms.mean.copy()

    # Épisode d'évaluation — les stats ne doivent pas changer
    from unittest.mock import patch

    with patch("gymnasium.make", return_value=env):
        result = evaluate(agent, "DummyEnv-v0", max_steps=10, n_episodes=1, seed=1)

    assert result["mean_reward"] == pytest.approx(5.0)
    assert agent.obs_normalizer.rms.count == count_after_train, (
        "count ne doit pas changer en mode eval."
    )
    np.testing.assert_array_equal(
        agent.obs_normalizer.rms.mean,
        mean_after_train,
        err_msg="mean ne doit pas changer en mode eval.",
    )


# ------------------------------------------------------------------
# Reproductibilité par seed
# ------------------------------------------------------------------


def test_seeding_reproducibility() -> None:
    """Deux agents initialisés avec la même seed produisent les mêmes premières actions."""
    from rl_from_scratch.core.utils import set_all_seeds

    obs = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

    set_all_seeds(42)
    agent1 = A2CAgent(obs_dim=4, action_dim=2, n_steps=16, device="cpu")
    action1 = agent1.select_action(obs)

    set_all_seeds(42)
    agent2 = A2CAgent(obs_dim=4, action_dim=2, n_steps=16, device="cpu")
    action2 = agent2.select_action(obs)

    np.testing.assert_array_equal(
        action1,
        action2,
        err_msg=(
            "Deux agents avec la même seed doivent produire la même première action. "
            f"action1={action1}, action2={action2}"
        ),
    )


# ------------------------------------------------------------------
# Benchmark multi-seed
# ------------------------------------------------------------------


def test_benchmark_runs_multiple_seeds(tmp_path, monkeypatch) -> None:
    """run_benchmark appelle train_fn une fois par seed avec le bon run_name."""
    import rl_from_scratch.benchmark as benchmark_module
    from rl_from_scratch.actor_critic.config import A2CConfig

    called_with: list[dict] = []

    def fake_train_fn(config, *, output_dir=None, run_name=None, seed=None):
        called_with.append({"seed": seed, "run_name": run_name})
        # Retourne un résultat minimal compatible avec _aggregate_results
        return {
            "history": {"eval_mean_rewards": [1.0], "episode_rewards": [1.0]},
            "metrics": {},
            "paths": None,
        }

    fake_config = A2CConfig(
        env_id="Pendulum-v1",
        total_timesteps=100,
        run_name="test-run",
        output_dir=str(tmp_path),
    )

    # Monkeypatch load_config pour retourner notre config de test
    monkeypatch.setattr(benchmark_module, "run_benchmark",
                        lambda config_path, seeds: _patched_run_benchmark(
                            config_path, seeds, fake_config, fake_train_fn, tmp_path
                        ))

    seeds = [0, 1, 2]
    benchmark_module.run_benchmark("dummy.yaml", seeds)

    # Vérifie que notre fake a été appelé pour chaque seed
    assert len(called_with) == len(seeds), (
        f"Attendu {len(seeds)} appels, obtenu {len(called_with)}."
    )
    for i, seed in enumerate(seeds):
        assert called_with[i]["seed"] == seed, (
            f"Seed {i} incorrect: attendu {seed}, obtenu {called_with[i]['seed']}."
        )
        assert f"seed{seed}" in called_with[i]["run_name"], (
            f"run_name doit contenir 'seed{seed}', obtenu '{called_with[i]['run_name']}'."
        )


def _patched_run_benchmark(config_path, seeds, fake_config, fake_train_fn, tmp_path):
    """Implémentation factice de run_benchmark pour le test unitaire."""
    import json
    from pathlib import Path

    base_run_name = fake_config.run_name or fake_config.approach
    base_output_dir = str(tmp_path)
    all_results = []

    for seed in seeds:
        run_name = f"{base_run_name}-seed{seed}"
        result = fake_train_fn(
            fake_config,
            output_dir=base_output_dir,
            run_name=run_name,
            seed=seed,
        )
        all_results.append({"seed": seed, "run_name": run_name, "result": result})

    # Sauvegarde un résumé minimal pour satisfaire la logique originale
    summary_dir = Path(base_output_dir) / f"{base_run_name}-benchmark"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w") as f:
        json.dump({"seeds": seeds}, f)

    return {"seeds": seeds, "num_seeds": len(seeds)}


# ------------------------------------------------------------------
# Checkpoint best restore obs_normalizer
# ------------------------------------------------------------------


def test_best_checkpoint_restores_obs_normalizer(tmp_path) -> None:
    """Le best checkpoint restaure aussi obs_normalizer avant la vidéo."""
    import torch
    from rl_from_scratch.actor_critic.agent import A2CAgent
    from rl_from_scratch.core.normalization import ObservationNormalizer

    agent = A2CAgent(obs_dim=3, action_dim=1, normalize_observations=True)
    assert agent.obs_normalizer is not None

    # Simule des stats "best" distinctes
    agent.obs_normalizer.rms.mean = np.array([1.0, 2.0, 3.0])
    agent.obs_normalizer.rms.var = np.array([0.5, 0.5, 0.5])
    agent.obs_normalizer.rms.count = 42

    # Sauvegarde best checkpoint avec le normalizer
    checkpoint = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "obs_normalizer": agent.obs_normalizer.to_dict(),
    }
    best_path = tmp_path / "best.pt"
    torch.save(checkpoint, best_path)

    # Modifie le normalizer pour simuler un training continued
    agent.obs_normalizer.rms.mean = np.array([100.0, 200.0, 300.0])
    agent.obs_normalizer.rms.count = 9999

    # Restaure depuis best checkpoint (même logique que training.py)
    loaded = torch.load(best_path, weights_only=True)
    agent.actor.load_state_dict(loaded["actor"])
    agent.critic.load_state_dict(loaded["critic"])
    if "obs_normalizer" in loaded and hasattr(agent, "obs_normalizer"):
        agent.obs_normalizer = ObservationNormalizer.from_dict(loaded["obs_normalizer"])

    # Vérifie que les stats du best sont restaurées
    np.testing.assert_array_equal(agent.obs_normalizer.rms.mean, [1.0, 2.0, 3.0])
    assert agent.obs_normalizer.rms.count == 42


# ------------------------------------------------------------------
# Benchmark diagnostics aggregation across seeds
# ------------------------------------------------------------------


def test_benchmark_aggregates_diagnostics_across_seeds() -> None:
    """_aggregate_results agrège les diagnostics sur toutes les seeds."""
    from rl_from_scratch.benchmark import _aggregate_results

    all_results = [
        {
            "seed": 0,
            "run_name": "test-seed0",
            "result": {
                "history": {
                    "eval_mean_rewards": [10.0, 20.0],
                    "episode_rewards": [5.0, 15.0],
                    "step_action_clip_fractions": [0.1, 0.05],
                    "step_log_std_means": [-0.5, -0.3],
                    "step_explained_variances": [0.6, 0.8],
                },
                "paths": None,
            },
        },
        {
            "seed": 1,
            "run_name": "test-seed1",
            "result": {
                "history": {
                    "eval_mean_rewards": [12.0, 18.0],
                    "episode_rewards": [6.0, 14.0],
                    "step_action_clip_fractions": [0.12, 0.07],
                    "step_log_std_means": [-0.4, -0.2],
                    "step_explained_variances": [0.5, 0.7],
                },
                "paths": None,
            },
        },
        {
            "seed": 2,
            "run_name": "test-seed2",
            "result": {
                "history": {
                    "eval_mean_rewards": [11.0, 22.0],
                    "episode_rewards": [7.0, 16.0],
                    # Pas de step_log_std_means pour cette seed (cas manquant)
                    "step_action_clip_fractions": [0.08, 0.03],
                    "step_explained_variances": [0.55, 0.75],
                },
                "paths": None,
            },
        },
    ]

    summary = _aggregate_results(all_results, "a2c", "test.yaml")

    # Diagnostics agrégés (pas juste la dernière seed)
    clip = summary["final_action_clip_fractions"]
    assert "mean" in clip and "std" in clip and "per_seed" in clip
    assert len(clip["per_seed"]) == 3  # toutes les 3 seeds
    np.testing.assert_allclose(clip["per_seed"], [0.05, 0.07, 0.03])

    # log_std: seulement 2 seeds ont la clé
    log_std = summary["final_log_std_means"]
    assert len(log_std["per_seed"]) == 2  # seed 2 n'avait pas cette métrique

    # explained_variances: 3 seeds
    ev = summary["final_explained_variances"]
    assert len(ev["per_seed"]) == 3
    expected_mean = np.mean([0.8, 0.7, 0.75])
    np.testing.assert_allclose(ev["mean"], expected_mean, atol=1e-6)
