from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_from_scratch.actor_critic.reporting import ActorCriticReporting
from rl_from_scratch.core.normalization import ObservationNormalizer
from rl_from_scratch.core.reporting import _clip_action, _maybe_normalize, record_greedy_episode
from rl_from_scratch.trust_region.reporting import TrustRegionReporting


class DummyAgent:
    def select_action(self, observation, *, deterministic=False):
        assert deterministic is True
        return 0


class DummyEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(seed=lambda seed: None)
        self.step_count = 0

    def reset(self, seed=None):
        return [0.0], {}

    def step(self, action):
        self.step_count += 1
        return [0.0], 1.0, True, False, {}

    def close(self):
        pass


class DummyRecordVideo:
    kwargs = None

    def __init__(self, env, **kwargs):
        DummyRecordVideo.kwargs = kwargs
        self.env = env

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class DummyRecordVideoWrites:
    kwargs = None

    def __init__(self, env, **kwargs):
        DummyRecordVideoWrites.kwargs = kwargs
        self.env = env

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        folder = Path(self.kwargs["video_folder"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.kwargs['name_prefix']}-episode-0.mp4").touch()
        self.env.close()


def test_record_greedy_episode_passes_record_video_options(monkeypatch, tmp_path):
    made_envs = []

    def fake_make(env_id, render_mode=None, **kwargs):
        made_envs.append((env_id, render_mode, kwargs))
        return DummyEnv()

    monkeypatch.setattr("gymnasium.make", fake_make)
    monkeypatch.setattr("gymnasium.wrappers.RecordVideo", DummyRecordVideo)

    config = SimpleNamespace(
        env_id="CartPole-v1",
        seed=123,
        max_steps_per_episode=5,
    )
    video_dir = tmp_path / "videos"
    expected_video = video_dir / "greedy-episode-episode-0.mp4"
    video_dir.mkdir()
    expected_video.touch()

    result = record_greedy_episode(
        DummyAgent(),
        config,
        tmp_path,
        video_dir=video_dir,
        render_mode="rgb_array",
        episode_trigger=lambda episode: episode == 0,
        step_trigger=None,
        video_length=10,
        name_prefix="greedy-episode",
        fps=30,
        disable_logger=False,
        env_kwargs={"exclude_current_positions_from_observation": False},
    )

    assert made_envs == [
        (
            "CartPole-v1",
            "rgb_array",
            {"exclude_current_positions_from_observation": False},
        )
    ]
    assert DummyRecordVideo.kwargs["video_folder"] == str(video_dir)
    assert DummyRecordVideo.kwargs["video_length"] == 10
    assert DummyRecordVideo.kwargs["name_prefix"] == "greedy-episode"
    assert DummyRecordVideo.kwargs["fps"] == 30
    assert DummyRecordVideo.kwargs["disable_logger"] is False
    assert DummyRecordVideo.kwargs["episode_trigger"](0) is True
    assert result == expected_video


def test_record_greedy_episode_uses_unique_default_video_subdir(monkeypatch, tmp_path):
    def fake_make(env_id, render_mode=None, **kwargs):
        return DummyEnv()

    monkeypatch.setattr("gymnasium.make", fake_make)
    monkeypatch.setattr("gymnasium.wrappers.RecordVideo", DummyRecordVideoWrites)
    monkeypatch.setattr("time.time_ns", lambda: 123456)

    config = SimpleNamespace(
        env_id="CartPole-v1",
        seed=123,
        max_steps_per_episode=5,
    )

    result = record_greedy_episode(DummyAgent(), config, tmp_path)

    expected_dir = tmp_path / "figures" / "videos" / "greedy-episode-123456"
    assert DummyRecordVideoWrites.kwargs["video_folder"] == str(expected_dir)
    assert result == expected_dir / "greedy-episode-episode-0.mp4"


def test_record_greedy_episode_rejects_human_render_mode(tmp_path):
    config = SimpleNamespace(
        env_id="CartPole-v1",
        seed=123,
        max_steps_per_episode=5,
    )

    with pytest.raises(ValueError, match="render_mode='human'"):
        record_greedy_episode(
            DummyAgent(),
            config,
            tmp_path,
            render_mode="human",
        )


def test_actor_critic_reporting_keeps_old_history_compatible(tmp_path):
    config = SimpleNamespace(approach="a2c", env_id="CartPole-v1")
    history = {
        "episode_rewards": [1.0, 2.0, 3.0],
        "step_policy_losses": [0.3, 0.2],
        "step_value_losses": [0.6, 0.5],
        "step_total_losses": [0.9, 0.7],
        "step_entropies": [0.1, 0.09],
    }

    figures = ActorCriticReporting().generate_figures(history, config, tmp_path)
    names = {path.name for path in figures}

    assert "learning_curves.png" in names
    assert "loss_curves.png" in names
    assert "policy_loss.png" in names
    assert "value_loss.png" in names
    assert "total_loss.png" in names
    assert "entropy.png" in names
    assert "eval_diagnostics.png" not in names


def test_actor_critic_reporting_adds_optional_diagnostics_when_present(tmp_path):
    config = SimpleNamespace(approach="a2c_gae", env_id="HalfCheetah-v5")
    history = {
        "episode_rewards": [10.0, 12.0, 11.0, 13.0],
        "eval_steps": [2, 4],
        "eval_mean_rewards": [100.0, 95.0],
        "eval_std_rewards": [5.0, 7.0],
        "eval_min_rewards": [90.0, 84.0],
        "eval_max_rewards": [110.0, 106.0],
        "eval_success_rates": [0.25, 0.5],
        "step_explained_variances": [0.1, 0.4],
        "step_grad_norms": [0.8, 0.6],
        "step_log_std_means": [-0.1, -0.2],
        "step_action_clip_fractions": [0.05, 0.02],
    }

    figures = ActorCriticReporting().generate_figures(history, config, tmp_path)
    names = {path.name for path in figures}

    assert "eval_diagnostics.png" in names
    assert "explained_variance.png" in names
    assert "grad_norm.png" in names
    assert "log_std.png" in names
    assert "action_clipping.png" in names


def test_trust_region_reporting_adds_optional_diagnostics_when_present(tmp_path):
    config = SimpleNamespace(approach="ppo", env_id="HalfCheetah-v5", target_kl=0.01)
    history = {
        "episode_rewards": [10.0, 12.0, 11.0, 13.0],
        "step_policy_losses": [0.3, 0.2],
        "step_value_losses": [0.6, 0.5],
        "step_kl": [0.001, 0.002],
        "step_entropies": [0.1, 0.09],
        "step_explained_variances": [0.1, 0.4],
        "step_grad_norms": [0.8, 0.6],
        "step_log_std_means": [-0.1, -0.2],
        "step_action_clip_fractions": [0.05, 0.02],
        "step_clip_fractions": [0.2, 0.1],
        "step_ratio_means": [1.0, 1.01],
        "step_ratio_stds": [0.03, 0.04],
        "step_line_search_accepts": [1.0, 0.0],
        "step_line_search_step_fractions": [1.0, 0.0],
    }

    figures = TrustRegionReporting().generate_figures(history, config, tmp_path)
    names = {path.name for path in figures}

    assert "kl_divergence.png" in names
    assert "explained_variance.png" in names
    assert "grad_norm.png" in names
    assert "log_std.png" in names
    assert "action_clipping.png" in names
    assert "ppo_clip_fraction.png" in names
    assert "ppo_ratio_mean.png" in names
    assert "ppo_ratio_std.png" in names
    assert "trpo_line_search_accept.png" in names
    assert "trpo_step_fraction.png" in names


# ── Tests pour _maybe_normalize et _clip_action dans record_greedy_episode ──


class NormalizerAgent:
    """Agent factice avec un normalizer dont on peut traquer les appels."""

    def __init__(self, obs_dim: int):
        self.obs_normalizer = ObservationNormalizer(obs_dim=obs_dim)
        # Pré-charge des stats réalistes
        self.obs_normalizer.rms.mean = np.array([10.0] * obs_dim)
        self.obs_normalizer.rms.var = np.array([4.0] * obs_dim)
        self.obs_normalizer.rms.count = 100

    def select_action(self, observation, *, deterministic=False):
        return 0


def test_maybe_normalize_applies_normalizer_without_update():
    """_maybe_normalize applique le normalizer sans mettre à jour les stats."""
    agent = NormalizerAgent(obs_dim=3)
    count_before = agent.obs_normalizer.rms.count

    obs = np.array([12.0, 8.0, 10.0], dtype=np.float32)
    result = _maybe_normalize(agent, obs)

    # Stats non modifiées (update=False)
    assert agent.obs_normalizer.rms.count == count_before
    # L'obs a été normalisée : (obs - mean) / sqrt(var + eps)
    expected = (obs - agent.obs_normalizer.rms.mean) / np.sqrt(
        agent.obs_normalizer.rms.var + agent.obs_normalizer.epsilon
    )
    np.testing.assert_allclose(result, expected.astype(np.float32), atol=1e-6)


def test_maybe_normalize_passthrough_without_normalizer():
    """_maybe_normalize passe l'obs telle quelle si pas de normalizer."""
    agent = DummyAgent()  # pas de obs_normalizer
    obs = [1.0, 2.0, 3.0]
    result = _maybe_normalize(agent, obs)
    assert result is obs  # même objet, pas de copie


def test_clip_action_clips_continuous():
    """_clip_action clip aux bornes de l'action space continu."""
    env = SimpleNamespace(
        action_space=SimpleNamespace(
            low=np.array([-1.0, -2.0]),
            high=np.array([1.0, 2.0]),
        )
    )
    action = np.array([3.0, -5.0])
    clipped = _clip_action(action, env)
    np.testing.assert_array_equal(clipped, np.array([1.0, -2.0]))


def test_clip_action_passthrough_discrete():
    """_clip_action ne clip pas les actions discrètes (pas de .low)."""
    env = SimpleNamespace(action_space=SimpleNamespace(n=4))
    action = 2
    result = _clip_action(action, env)
    assert result == 2


def test_record_greedy_episode_normalizes_obs(monkeypatch, tmp_path):
    """record_greedy_episode normalise les observations via agent.obs_normalizer."""
    observed_states = []

    class TrackingAgent:
        def __init__(self):
            self.obs_normalizer = ObservationNormalizer(obs_dim=2)
            self.obs_normalizer.rms.mean = np.array([5.0, 5.0])
            self.obs_normalizer.rms.var = np.array([1.0, 1.0])
            self.obs_normalizer.rms.count = 50

        def select_action(self, observation, *, deterministic=False):
            observed_states.append(np.array(observation))
            return 0

    class NormEnv:
        def __init__(self):
            self.action_space = SimpleNamespace(seed=lambda s: None)

        def reset(self, seed=None):
            return np.array([7.0, 3.0], dtype=np.float32), {}

        def step(self, action):
            return np.array([7.0, 3.0], dtype=np.float32), 1.0, True, False, {}

        def close(self):
            pass

    monkeypatch.setattr("gymnasium.make", lambda *a, **kw: NormEnv())
    monkeypatch.setattr("gymnasium.wrappers.RecordVideo", DummyRecordVideo)

    agent = TrackingAgent()
    count_before = agent.obs_normalizer.rms.count
    config = SimpleNamespace(env_id="Test-v0", seed=0, max_steps_per_episode=3)

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "greedy-episode-episode-0.mp4").touch()

    record_greedy_episode(agent, config, tmp_path, video_dir=video_dir)

    # L'agent a vu des observations normalisées, pas brutes
    assert len(observed_states) >= 1
    raw = np.array([7.0, 3.0])
    for state in observed_states:
        # Si brut, state serait [7, 3]. Normalisé, c'est ~[2, -2]
        assert not np.allclose(state, raw), "L'obs n'a pas été normalisée"

    # Les stats n'ont pas été mises à jour
    assert agent.obs_normalizer.rms.count == count_before
