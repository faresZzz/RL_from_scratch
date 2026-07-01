"""Protocol tests for the multi-seed benchmark and off-policy evaluation cadence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pytest

import rl_from_scratch.core.config as config_module
from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.recording import RunManager, RunRecorder
from rl_from_scratch.benchmark import _aggregate_results, run_benchmark
from rl_from_scratch.deterministic_actor_critic.config import DDPGConfig, TD3Config
from rl_from_scratch.deterministic_actor_critic.training import train_ddpg, train_td3
import rl_from_scratch.deterministic_actor_critic.training as dac_training
from rl_from_scratch.sac.config import SACConfig
from rl_from_scratch.sac.training import train_sac
import rl_from_scratch.sac.training as sac_training


def _benchmark_result(
    *,
    run_dir: Path | str,
    eval_mean_rewards: list[float],
    eval_steps: list[int],
    eval_timesteps: list[int] | None = None,
    episode_rewards: list[float] | None = None,
) -> dict[str, Any]:
    """Build the minimal training result shape consumed by benchmark aggregation."""
    return {
        "history": {
            "eval_mean_rewards": eval_mean_rewards,
            "eval_steps": eval_steps,
            "eval_timesteps": eval_timesteps or [],
            "episode_rewards": episode_rewards or eval_mean_rewards,
        },
        "metrics": {},
        "paths": {"run_dir": str(run_dir)},
    }


def _patch_visual_artifacts(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """Disable figure/video generation for short training tests.

    record_greedy_episode is now patched globally via conftest autouse
    fixture (_disable_greedy_video).  Here we only disable per-package
    figure generation when the reporting module is available.
    """
    if hasattr(module, "_dac_reporting"):
        monkeypatch.setattr(
            module._dac_reporting,
            "generate_training_figures",
            lambda *args, **kwargs: [],
        )
    if hasattr(module, "_sac_reporting"):
        monkeypatch.setattr(
            module._sac_reporting,
            "generate_training_figures",
            lambda *args, **kwargs: [],
        )


def _make_short_ddpg_config(seed: int = 7) -> DDPGConfig:
    return DDPGConfig(
        env_id="Pendulum-v1",
        seed=seed,
        total_timesteps=50,
        max_steps_per_episode=25,
        hidden_dim=32,
        batch_size=8,
        buffer_capacity=128,
        start_steps=8,
        update_after=8,
        checkpoint_every=10_000,
        eval_every=1,
        eval_episodes=1,
        device="cpu",
    )


def _make_short_td3_config(seed: int = 7) -> TD3Config:
    return TD3Config(
        env_id="Pendulum-v1",
        seed=seed,
        total_timesteps=50,
        max_steps_per_episode=25,
        hidden_dim=32,
        batch_size=8,
        buffer_capacity=128,
        start_steps=8,
        update_after=8,
        checkpoint_every=10_000,
        eval_every=1,
        eval_episodes=1,
        policy_delay=2,
        device="cpu",
    )


def _make_short_sac_config(seed: int = 7) -> SACConfig:
    return SACConfig(
        env_id="Pendulum-v1",
        seed=seed,
        total_timesteps=50,
        max_steps_per_episode=25,
        hidden_dim=32,
        batch_size=8,
        buffer_capacity=128,
        start_steps=8,
        update_after=8,
        checkpoint_every=10_000,
        eval_every=1,
        eval_episodes=1,
        device="cpu",
    )


def _patch_fake_offpolicy_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    tmp_path: Path,
    train_lengths: list[int],
    eval_mean_reward: float = 123.0,
) -> dict[str, list[int]]:
    """Replace env/training side effects with a deterministic lightweight harness.

    The new training modules inline the episode loop (no separate
    ``train_one_episode``).  This harness provides realistic env/agent fakes
    that the inline loop can drive, with controllable episode lengths and
    seed tracking.
    """
    state: dict[str, list[int]] = {"train_seeds": [], "eval_seeds": []}
    remaining_lengths = list(train_lengths)

    class _FakeEnv:
        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def __init__(self) -> None:
            self._step_count = 0
            self._target_length = 1

        def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
            state["train_seeds"].append(seed)
            self._step_count = 0
            self._target_length = remaining_lengths.pop(0) if remaining_lengths else 1
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            self._step_count += 1
            done = self._step_count >= self._target_length
            return np.zeros(3, dtype=np.float32), 1.0, False, done, {}

        def close(self) -> None:
            return None

    class _FakeNoise:
        def reset(self) -> None:
            pass

    class _FakeAgent:
        _last_raw_action = np.zeros(1, dtype=np.float32)
        noise = _FakeNoise()

        def select_action(self, obs: Any, deterministic: bool = False) -> np.ndarray:
            a = np.zeros(1, dtype=np.float32)
            self._last_raw_action = a
            return a

        def store_transition(self, *args: Any, **kwargs: Any) -> None:
            pass

        def learn_step(self) -> dict[str, float]:
            return {"critic_loss": 1.0, "actor_loss": 0.5}

        def episode_ended(self) -> None:
            pass

        def record_action_diagnostics(self, **kwargs: Any) -> None:
            pass

        def save(self, path: Any) -> None:
            pass

        def load(self, path: Any) -> None:
            pass

    def _fake_evaluate(
        agent: Any,
        env_id: str,
        *,
        n_episodes: int,
        seed: int,
        max_steps: int,
        solved_reward: float | None = None,
    ) -> dict[str, float]:
        state["eval_seeds"].append(seed)
        return {
            "mean_reward": eval_mean_reward,
            "std_reward": 0.0,
            "min_reward": eval_mean_reward,
            "max_reward": eval_mean_reward,
            "mean_length": 1.0,
        }

    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: _FakeEnv())
    monkeypatch.setattr(
        module,
        "get_env_info",
        lambda env: {"obs_dim": 3, "action_dim": 1},
    )
    monkeypatch.setattr(
        module,
        "build_agent",
        lambda *args, **kwargs: _FakeAgent(),
    )
    # Patch evaluate on the module — RunManager.from_config stores the
    # module-level reference as evaluate_fn, so this intercept works.
    monkeypatch.setattr(module, "evaluate", _fake_evaluate)
    # record_greedy_episode is patched globally via conftest autouse fixture.
    monkeypatch.setattr(RunManager, "checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        RunManager,
        "finish",
        lambda self, **kwargs: self.recorder.finalize(**kwargs),
    )
    monkeypatch.setattr(
        RunRecorder,
        "persist",
        lambda self, paths, **kwargs: self.finalize(**kwargs),
    )
    if hasattr(module, "_dac_reporting"):
        monkeypatch.setattr(
            module._dac_reporting,
            "generate_training_figures",
            lambda *args, **kwargs: [],
        )
    if hasattr(module, "_sac_reporting"):
        monkeypatch.setattr(
            module._sac_reporting,
            "generate_training_figures",
            lambda *args, **kwargs: [],
        )

    return state


def test_base_config_protocol_fields_and_validation() -> None:
    """BaseConfig exposes num_seeds in serialization and validates it strictly."""
    config = BaseConfig(
        env_id="Pendulum-v1",
        approach="protocol_test",
        total_timesteps=32,
        eval_every=4,
        eval_episodes=2,
        num_seeds=3,
    )

    payload = config.to_dict()

    assert payload["num_seeds"] == 3
    assert BaseConfig.from_dict(payload).num_seeds == 3

    with pytest.raises(ValueError, match="num_seeds must be at least 1"):
        BaseConfig(num_seeds=0)


def test_run_benchmark_uses_config_num_seeds_when_seeds_is_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When seeds=None, benchmark should derive seeds from config.num_seeds."""
    config = BaseConfig(
        env_id="Pendulum-v1",
        approach="protocol_benchmark",
        total_timesteps=10,
        run_name="proto",
        output_dir=str(tmp_path),
        num_seeds=3,
    )
    called_seeds: list[int] = []

    def _fake_train(
        cfg: BaseConfig,
        *,
        output_dir: str | None = None,
        run_name: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        assert seed is not None
        called_seeds.append(seed)
        return _benchmark_result(
            run_dir=tmp_path / f"seed{seed}",
            eval_mean_rewards=[10.0 + seed],
            eval_steps=[100],
        )

    monkeypatch.setattr(config_module, "load_config", lambda _: config)
    monkeypatch.setitem(config_module.AGENT_FACTORIES, "protocol_benchmark", _fake_train)

    summary = run_benchmark("ignored.yaml", seeds=None)

    assert called_seeds == [0, 1, 2]
    assert summary["seeds"] == [0, 1, 2]
    assert summary["num_seeds"] == 3


def test_run_benchmark_rejects_seed_without_eval() -> None:
    """Aggregation should fail fast when one seed produced no evaluation trajectory."""
    entry = {
        "seed": 1,
        "run_name": "proto-seed1",
        "result": _benchmark_result(
            run_dir="run1",
            eval_mean_rewards=[],
            eval_steps=[],
            episode_rewards=[1.0, 2.0],
        ),
    }

    with pytest.raises(ValueError, match=r"Seed 1.*evaluation"):
        from rl_from_scratch.benchmark import _require_deterministic_eval

        _require_deterministic_eval(entry)


def test_run_benchmark_writes_a_unique_summary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each benchmark should persist one summary in its own unique directory."""
    config = BaseConfig(
        env_id="Pendulum-v1",
        approach="protocol_summary",
        total_timesteps=10,
        run_name="proto",
        output_dir=str(tmp_path),
        num_seeds=2,
    )

    def _fake_train(
        cfg: BaseConfig,
        *,
        output_dir: str | None = None,
        run_name: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        assert seed is not None
        return _benchmark_result(
            run_dir=tmp_path / f"seed{seed}",
            eval_mean_rewards=[float(seed)],
            eval_steps=[25],
        )

    monkeypatch.setattr(config_module, "load_config", lambda _: config)
    monkeypatch.setitem(config_module.AGENT_FACTORIES, "protocol_summary", _fake_train)

    run_benchmark("ignored.yaml", seeds=[2, 4])

    summary_paths = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("summary.json")
    )
    assert len(summary_paths) == 1
    assert summary_paths[0].startswith("proto-benchmark-")
    assert summary_paths[0].endswith("/summary.json")


def test_aggregate_results_exposes_contributing_seeds_and_final_eval_timestep() -> None:
    """Final eval aggregation should retain both contributing seed ids and the timestep."""
    summary = _aggregate_results(
        [
            {
                "seed": 3,
                "run_name": "proto-seed3",
                "result": _benchmark_result(
                    run_dir="run3",
                    eval_mean_rewards=[5.0, 7.0],
                    eval_steps=[50, 100],
                    eval_timesteps=[50_000, 100_000],
                ),
            },
            {
                "seed": 7,
                "run_name": "proto-seed7",
                "result": _benchmark_result(
                    run_dir="run7",
                    eval_mean_rewards=[6.0, 9.0],
                    eval_steps=[50, 100],
                    eval_timesteps=[50_000, 100_000],
                ),
            },
        ],
        "sac",
        "protocol.yaml",
    )

    final_eval = summary["final_eval_mean_reward"]
    final_timestep = summary["final_eval_timestep"]
    assert final_eval["contributing_seeds"] == [3, 7]
    assert final_timestep["contributing_seeds"] == [3, 7]
    np.testing.assert_allclose(final_eval["per_seed"], [7.0, 9.0])
    np.testing.assert_allclose(final_timestep["per_seed"], [100_000.0, 100_000.0])


@pytest.mark.parametrize(
    ("label", "train_fn", "config_factory", "module"),
    [
        ("ddpg", train_ddpg, _make_short_ddpg_config, dac_training),
        ("td3", train_td3, _make_short_td3_config, dac_training),
        ("sac", train_sac, _make_short_sac_config, sac_training),
    ],
)
def test_same_seed_cpu_training_is_reproducible(
    label: str,
    train_fn: Callable[..., dict[str, Any]],
    config_factory: Callable[[int], BaseConfig],
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DDPG/TD3/SAC should be reproducible on CPU when rerun with the same seed."""
    _patch_visual_artifacts(monkeypatch, module)
    config = config_factory(11)

    result_a = train_fn(config, output_dir=str(tmp_path / f"{label}-a"), seed=11)
    result_b = train_fn(config, output_dir=str(tmp_path / f"{label}-b"), seed=11)

    assert result_a["history"]["episode_lengths"] == result_b["history"]["episode_lengths"]
    np.testing.assert_allclose(
        result_a["history"]["episode_rewards"],
        result_b["history"]["episode_rewards"],
    )
    np.testing.assert_allclose(
        result_a["history"]["eval_mean_rewards"],
        result_b["history"]["eval_mean_rewards"],
    )


@pytest.mark.parametrize(
    ("train_fn", "config_factory", "module"),
    [
        (train_ddpg, _make_short_ddpg_config, dac_training),
        (train_sac, _make_short_sac_config, sac_training),
    ],
)
def test_eval_seed_is_fixed_and_independent_from_training_episode_seed(
    train_fn: Callable[..., dict[str, Any]],
    config_factory: Callable[[int], BaseConfig],
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation should always reuse the base seed, even as training episode seeds advance."""
    state = _patch_fake_offpolicy_runtime(
        monkeypatch,
        module,
        tmp_path=tmp_path,
        train_lengths=[10, 10],
    )
    config = config_factory(17)
    config.total_timesteps = 20
    config.eval_every = 1
    config.eval_episodes = 1
    config.eval_seed = 999

    train_fn(config, output_dir=str(tmp_path), seed=17)

    assert state["train_seeds"] == [17, 18]
    assert state["eval_seeds"] == [999, 999]


@pytest.mark.parametrize(
    ("train_fn", "config_factory", "module"),
    [
        (train_ddpg, _make_short_ddpg_config, dac_training),
        (train_sac, _make_short_sac_config, sac_training),
    ],
)
def test_offpolicy_short_loops_report_exact_observed_timesteps(
    train_fn: Callable[..., dict[str, Any]],
    config_factory: Callable[[int], BaseConfig],
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short episodic DDPG/SAC loops should stop on the exact requested timestep budget."""
    _patch_fake_offpolicy_runtime(
        monkeypatch,
        module,
        tmp_path=tmp_path,
        train_lengths=[7, 7, 6],
    )
    config = config_factory(23)
    config.total_timesteps = 20
    config.eval_every = 99

    result = train_fn(config, output_dir=str(tmp_path), seed=23)

    assert result["metrics"]["observed_timesteps"] == 20
    assert result["metrics"]["total_timesteps"] == 20


@pytest.mark.parametrize(
    ("train_fn", "config_factory", "module"),
    [
        (train_ddpg, _make_short_ddpg_config, dac_training),
        (train_sac, _make_short_sac_config, sac_training),
    ],
)
def test_offpolicy_eval_cadence_is_environment_step_based_and_includes_final_eval(
    train_fn: Callable[..., dict[str, Any]],
    config_factory: Callable[[int], BaseConfig],
    module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off-policy protocol should evaluate on env-step cadence and always include the final step."""
    _patch_fake_offpolicy_runtime(
        monkeypatch,
        module,
        tmp_path=tmp_path,
        train_lengths=[7, 7, 6],
    )
    config = config_factory(29)
    config.total_timesteps = 20
    config.eval_every = 10
    config.eval_episodes = 1
    config.eval_every_steps = 10

    result = train_fn(config, output_dir=str(tmp_path), seed=29)

    assert result["history"]["eval_timesteps"] == [14, 20]
    assert result["metrics"]["final_eval_timestep"] == 20
