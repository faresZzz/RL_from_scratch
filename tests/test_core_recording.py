from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rl_from_scratch.core.artifacts import ExperimentPaths
from rl_from_scratch.core.recording import RunManager, RunRecorder, record_policy_video


class DummyAgent:
    def save(self, path: str | Path) -> None:
        Path(path).write_text(f"saved:{Path(path).name}", encoding="utf-8")


class DummyRgbEnv:
    def __init__(self) -> None:
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = SimpleNamespace(low=-1.0, high=1.0, shape=())
        self.reset_seeds: list[int | None] = []
        self.closed = False

    def reset(self, seed: int | None = None):
        self.reset_seeds.append(seed)
        return 0.0, {}

    def step(self, action):
        return 0.0, 1.0, True, False, {}

    def close(self) -> None:
        self.closed = True


class DummyRecordVideoWrites:
    kwargs = None
    env = None

    def __init__(self, env, **kwargs):
        DummyRecordVideoWrites.kwargs = kwargs
        DummyRecordVideoWrites.env = env
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        folder = Path(self.kwargs["video_folder"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.kwargs['name_prefix']}-episode-0.mp4").touch()
        self.env.close()


def test_record_policy_video_uses_rgb_array_record_video_and_closes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    made_envs = []

    def fake_make(env_id, render_mode=None, **kwargs):
        env = DummyRgbEnv()
        made_envs.append((env_id, render_mode, kwargs, env))
        return env

    monkeypatch.setattr("gymnasium.make", fake_make)
    monkeypatch.setattr("gymnasium.wrappers.RecordVideo", DummyRecordVideoWrites)

    videos = record_policy_video(
        "CartPole-v1",
        lambda obs: 0,
        video_dir=tmp_path / "videos",
        episodes=1,
        seed=123,
        max_steps=5,
        name_prefix="policy-demo",
        env_kwargs={"foo": "bar"},
    )

    assert made_envs[0][:3] == ("CartPole-v1", "rgb_array", {"foo": "bar"})
    assert made_envs[0][3].closed is True
    assert DummyRecordVideoWrites.kwargs["video_folder"] == str(tmp_path / "videos")
    assert DummyRecordVideoWrites.kwargs["name_prefix"] == "policy-demo"
    assert videos == [tmp_path / "videos" / "policy-demo-episode-0.mp4"]


def test_run_recorder_builds_flat_history_and_summary() -> None:
    recorder = RunRecorder()

    recorder.record_episode(reward=10.0, length=5, epsilon=0.4)
    recorder.record_episode(reward=30.0, length=7, epsilon=0.2)
    recorder.record_evaluation(
        step=2,
        timestep=120,
        mean_reward=25.0,
        std_reward=2.0,
        min_reward=23.0,
        max_reward=27.0,
        success_rate=0.5,
    )
    recorder.record_evaluation(
        step=3,
        timestep=180,
        mean_reward=40.0,
        std_reward=1.5,
        min_reward=39.0,
        max_reward=42.0,
        success_rate=1.0,
    )
    recorder.record_update({"policy_loss": 0.9, "value_loss": 1.2})
    recorder.record_update({"policy_loss": 0.7, "entropy": 0.1})

    result = recorder.finalize(
        total_timesteps=1_000,
        observed_timesteps=180,
        episodes_to_solve=2,
    )

    history = result["history"]
    metrics = result["metrics"]

    assert history["episode_rewards"] == [10.0, 30.0]
    assert history["episode_lengths"] == [5, 7]
    assert history["epsilons"] == [0.4, 0.2]
    assert history["eval_steps"] == [2, 3]
    assert history["eval_timesteps"] == [120, 180]
    assert history["eval_mean_rewards"] == [25.0, 40.0]
    assert history["step_policy_losses"] == [0.9, 0.7]
    assert history["step_value_losses"] == [1.2]
    assert history["step_entropies"] == [0.1]

    assert metrics["episodes"] == 2
    assert metrics["total_timesteps"] == 1_000
    assert metrics["observed_timesteps"] == 180
    assert metrics["mean_reward"] == 20.0
    assert metrics["best_reward"] == 30.0
    assert metrics["best_eval_mean_reward"] == 40.0
    assert metrics["best_eval_timestep"] == 180
    assert metrics["final_eval_mean_reward"] == 40.0
    assert metrics["episodes_to_solve"] == 2


def test_run_recorder_persist_writes_history_and_metrics(tmp_path: Path) -> None:
    recorder = RunRecorder()
    recorder.record_episode(reward=5.0, length=3)

    paths = ExperimentPaths(
        run_dir=tmp_path,
        figure_dir=tmp_path / "figures",
        checkpoint_dir=tmp_path / "checkpoints",
        history_path=tmp_path / "history.json",
        metrics_path=tmp_path / "metrics.json",
        config_path=tmp_path / "config.json",
    )

    recorder.persist(paths)

    assert paths.history_path.exists()
    assert paths.metrics_path.exists()


def test_run_recorder_filters_non_finite_update_metrics() -> None:
    recorder = RunRecorder()

    recorder.record_update(
        {
            "policy_loss": 0.9,
            "value_loss": float("nan"),
            "entropy": float("inf"),
            "kl": -float("inf"),
        }
    )

    assert recorder.history["step_policy_losses"] == [0.9]
    assert "step_value_losses" not in recorder.history
    assert "step_entropies" not in recorder.history
    assert "step_kl" not in recorder.history


def test_run_recorder_uses_action_jepa_phase_a_metric_aliases() -> None:
    recorder = RunRecorder()

    recorder.record_update(
        {
            "representation_prediction_loss": 0.4,
            "representation_variance_loss": 0.3,
            "representation_covariance_loss": 0.2,
            "effective_rank": 3.0,
        }
    )

    assert recorder.history["step_representation_prediction_losses"] == [0.4]
    assert recorder.history["step_representation_variance_losses"] == [0.3]
    assert recorder.history["step_representation_covariance_losses"] == [0.2]
    assert recorder.history["step_effective_ranks"] == [3.0]


def test_run_recorder_uses_model_based_metric_aliases() -> None:
    recorder = RunRecorder()

    recorder.record_update(
        {
            "recon_loss": 0.5,
            "reward_loss": 0.4,
            "model_loss": 0.9,
            "imagined_return": 1.2,
        }
    )

    assert recorder.history["step_recon_losses"] == [0.5]
    assert recorder.history["step_reward_losses"] == [0.4]
    assert recorder.history["step_model_losses"] == [0.9]
    assert recorder.history["step_imagined_returns"] == [1.2]
    assert "step_recon_losss" not in recorder.history
    assert "step_model_losss" not in recorder.history


def test_run_manager_records_history_and_tracks_best_eval(tmp_path: Path) -> None:
    manager = RunManager(paths=_paths(tmp_path))

    manager.record_episode(reward=10.0, length=4, epsilon=0.3)
    manager.record_updates({"policy_loss": 1.0, "value_loss": float("nan")})
    first_improved = manager.record_eval(
        step=1,
        timestep=100,
        mean_reward=12.0,
        std_reward=2.0,
        min_reward=9.0,
        max_reward=14.0,
    )
    second_improved = manager.record_eval(
        step=2,
        timestep=200,
        mean_reward=11.0,
        std_reward=1.0,
        min_reward=10.0,
        max_reward=12.0,
    )

    assert first_improved is True
    assert second_improved is False
    assert manager.best_eval_reward == 12.0
    assert manager.best_eval_step == 1
    assert manager.best_eval_timestep == 100
    assert manager.history["episode_rewards"] == [10.0]
    assert manager.history["epsilons"] == [0.3]
    assert manager.history["step_policy_losses"] == [1.0]
    assert "step_value_losses" not in manager.history


def test_run_manager_checkpoint_tracks_paths_prunes_and_copies_best(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manager = RunManager(paths=paths, agent=DummyAgent(), checkpoint_keep_last=2)

    manager.checkpoint(step=1, current_reward=10.0)
    manager.checkpoint(step=2, current_reward=5.0)
    newest_path = manager.checkpoint(step=3, current_reward=20.0)

    assert manager.checkpoint_paths == [
        paths.checkpoint_dir / "checkpoint_000001.pt",
        paths.checkpoint_dir / "checkpoint_000002.pt",
        newest_path,
    ]
    assert sorted(path.name for path in paths.checkpoint_dir.glob("checkpoint_*.pt")) == [
        "checkpoint_000002.pt",
        "checkpoint_000003.pt",
    ]
    assert (paths.checkpoint_dir / "best.pt").read_text(encoding="utf-8") == (
        "saved:checkpoint_000003.pt"
    )


def test_run_manager_finish_persists_recorder_contract(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manager = RunManager(paths=paths)
    manager.record_episode(reward=7.0, length=3)

    result = manager.finish(total_timesteps=30, observed_timesteps=21)

    assert result["history"]["episode_rewards"] == [7.0]
    assert result["metrics"]["total_timesteps"] == 30
    assert result["metrics"]["observed_timesteps"] == 21
    assert json.loads(paths.history_path.read_text(encoding="utf-8")) == result["history"]
    assert json.loads(paths.metrics_path.read_text(encoding="utf-8")) == result["metrics"]


def _paths(tmp_path: Path) -> ExperimentPaths:
    return ExperimentPaths(
        run_dir=tmp_path,
        figure_dir=tmp_path / "figures",
        checkpoint_dir=tmp_path / "checkpoints",
        history_path=tmp_path / "history.json",
        metrics_path=tmp_path / "metrics.json",
        config_path=tmp_path / "config.json",
    )
