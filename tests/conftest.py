from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class FailedLookup(Exception):
    pass


def import_public_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - failure message is the contract
        pytest.fail(
            f"Expected public module '{module_name}' to import successfully: "
            f"{exc.__class__.__name__}: {exc}"
        )


def get_public_attr(
    module: Any,
    *preferred_names: str,
    predicate=None,
    description: str = "public attribute",
):
    for name in preferred_names:
        if hasattr(module, name):
            candidate = getattr(module, name)
            if predicate is None or predicate(candidate):
                return candidate

    public_items = [
        (name, getattr(module, name))
        for name in dir(module)
        if not name.startswith("_")
    ]
    for name, candidate in public_items:
        if predicate is None or predicate(candidate):
            return candidate

    available = ", ".join(name for name, _ in public_items) or "<none>"
    pytest.fail(
        f"Expected {description} in module '{module.__name__}'. "
        f"Available public names: {available}"
    )


def call_with_supported_kwargs(callable_obj, **kwargs):
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return callable_obj(**kwargs)

    supported = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
    }
    return callable_obj(**supported)


def _normalize_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    pytest.fail(f"Could not read config values from {type(config)!r}")


def _lower_key_mapping(config: Any) -> dict[str, Any]:
    return {key.lower(): value for key, value in _normalize_mapping(config).items()}


def get_bin_counts(config: Any) -> tuple[int, int, int, int]:
    for name in ("bins", "num_bins", "state_bins"):
        if hasattr(config, name):
            value = tuple(int(item) for item in getattr(config, name))
            if len(value) == 4:
                return value

    lowered = _lower_key_mapping(config)
    for name in ("bins", "num_bins", "state_bins"):
        if name in lowered:
            value = tuple(int(item) for item in lowered[name])
            if len(value) == 4:
                return value

    field_groups = (
        (
            "position_bins",
            "velocity_bins",
            "angle_bins",
            "angle_velocity_bins",
        ),
        (
            "number_of_bin_position",
            "number_of_bin_velocity",
            "number_of_bin_angle",
            "number_of_bin_angle_velocity",
        ),
        (
            "cart_position_bins",
            "cart_velocity_bins",
            "pole_angle_bins",
            "pole_angle_velocity_bins",
        ),
    )

    for field_names in field_groups:
        try:
            return tuple(int(get_config_value(config, name)) for name in field_names)
        except FailedLookup:
            continue

    pytest.fail("Expected config to expose four CartPole discretization bin counts.")


def get_config_value(config: Any, *names: str) -> Any:
    lowered = _lower_key_mapping(config)
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
        lowered_name = name.lower()
        if lowered_name in lowered:
            return lowered[lowered_name]
    raise FailedLookup(", ".join(names))


def require_config_class():
    module = import_public_module("rl_from_scratch.tabular.config")
    return get_public_attr(
        module,
        "QLearningConfig",
        predicate=lambda candidate: inspect.isclass(candidate)
        and candidate.__name__.endswith("Config"),
        description="config class",
    )


def make_config(**overrides):
    config_class = require_config_class()
    return call_with_supported_kwargs(config_class, **overrides)


def require_discretizer_class():
    module = import_public_module("rl_from_scratch.tabular.discretization")
    return get_public_attr(
        module,
        "CartPoleDiscretizer",
        "ObservationDiscretizer",
        "Discretizer",
        predicate=inspect.isclass,
        description="discretizer class",
    )


def make_discretizer(*, config=None, bin_counts=None, low=None, high=None):
    discretizer_class = require_discretizer_class()
    config = config or make_config()
    bin_counts = bin_counts or get_bin_counts(config)
    low = low or (-4.8, -3.0, -0.418, -10.0)
    high = high or (4.8, 3.0, 0.418, 10.0)
    return call_with_supported_kwargs(
        discretizer_class,
        config=config,
        bins=bin_counts,
        bin_counts=bin_counts,
        num_bins=bin_counts,
        low=np.asarray(low, dtype=float),
        high=np.asarray(high, dtype=float),
        lower_bounds=np.asarray(low, dtype=float),
        upper_bounds=np.asarray(high, dtype=float),
    )


def discretize(discretizer, observation: Sequence[float]) -> tuple[int, ...]:
    method = get_public_attr(
        discretizer,
        "discretize",
        "digitize",
        "transform",
        predicate=callable,
        description="discretize method",
    )
    result = method(np.asarray(observation, dtype=float))
    return tuple(int(item) for item in result)


def require_agent_class():
    module = import_public_module("rl_from_scratch.tabular.agent")
    return get_public_attr(
        module,
        "QLearningAgent",
        predicate=inspect.isclass,
        description="QLearningAgent class",
    )


def make_agent(*, config=None, num_actions=2):
    agent_class = require_agent_class()
    config = config or make_config()
    state_shape = get_bin_counts(config)
    return call_with_supported_kwargs(
        agent_class,
        state_shape=state_shape,
        num_actions=num_actions,
        action_count=num_actions,
        action_space_n=num_actions,
        n_actions=num_actions,
        alpha=getattr(config, "alpha", 0.1),
        gamma=getattr(config, "gamma", 0.99),
        epsilon=getattr(config, "epsilon", 0.2),
        epsilon_decay=getattr(config, "epsilon_decay", 0.995),
        min_epsilon=getattr(config, "min_epsilon", 0.02),
    )


def get_q_table(agent) -> np.ndarray:
    if hasattr(agent, "q_table"):
        return np.asarray(agent.q_table)
    if hasattr(agent, "Q_matrix"):
        return np.asarray(agent.Q_matrix)
    pytest.fail("Expected QLearningAgent to expose a q_table array.")


def select_random_action(agent) -> int:
    method = get_public_attr(
        agent,
        "select_random_action",
        "random_action",
        "select_action_random",
        predicate=callable,
        description="random action selector",
    )
    return int(method())


def select_greedy_action(agent, state) -> int:
    method = get_public_attr(
        agent,
        "select_greedy_action",
        "greedy_action",
        "select_action_greedy",
        predicate=callable,
        description="greedy action selector",
    )
    return int(method(tuple(state)))


def select_epsilon_action(agent, state) -> int:
    if hasattr(agent, "select_action_epsilon_greedy"):
        return int(agent.select_action_epsilon_greedy(tuple(state)))

    method = get_public_attr(
        agent,
        "select_action",
        "act",
        predicate=callable,
        description="epsilon-greedy action selector",
    )
    return int(
        call_with_supported_kwargs(
            method,
            state=tuple(state),
            observation=tuple(state),
            epsilon=getattr(agent, "epsilon", None),
        )
    )


def call_learn(agent, *, state, action, reward, next_state, done):
    learn = get_public_attr(
        agent,
        "learn",
        predicate=callable,
        description="learn method",
    )
    kwargs = {
        "state": tuple(state),
        "action": action,
        "reward": reward,
        "next_state": tuple(next_state),
        "done": done,
    }
    signature = inspect.signature(learn)
    if {"reward", "state", "next_state", "action", "done"} <= set(signature.parameters):
        return learn(**kwargs)
    if {"state", "action", "reward", "next_state", "done"} <= set(signature.parameters):
        return learn(**kwargs)
    return learn(reward, tuple(state), tuple(next_state), action, done)


def require_training_entrypoint():
    module = import_public_module("rl_from_scratch.tabular.training")
    function = get_public_attr(
        module,
        "train_q_learning",
        "train",
        "run_training",
        predicate=callable,
        description="training entrypoint",
    )
    return module, function


def require_sarsa_agent_class():
    module = import_public_module("rl_from_scratch.tabular.agent")
    return get_public_attr(
        module,
        "SarsaAgent",
        predicate=inspect.isclass,
        description="SarsaAgent class",
    )


def make_sarsa_agent(*, config=None, num_actions=2):
    agent_class = require_sarsa_agent_class()
    config = config or make_config()
    state_shape = get_bin_counts(config)
    return call_with_supported_kwargs(
        agent_class,
        state_shape=state_shape,
        num_actions=num_actions,
        action_count=num_actions,
        action_space_n=num_actions,
        n_actions=num_actions,
        alpha=getattr(config, "alpha", 0.1),
        gamma=getattr(config, "gamma", 0.99),
        epsilon=getattr(config, "epsilon", 0.2),
        epsilon_decay=getattr(config, "epsilon_decay", 0.995),
        min_epsilon=getattr(config, "min_epsilon", 0.02),
    )


def call_sarsa_learn(agent, *, state, action, reward, next_state, next_action, done):
    learn = get_public_attr(
        agent,
        "learn",
        predicate=callable,
        description="learn method",
    )
    return learn(
        state=tuple(state),
        action=action,
        reward=reward,
        next_state=tuple(next_state),
        next_action=next_action,
        done=done,
    )


def require_sarsa_config_class():
    module = import_public_module("rl_from_scratch.tabular.config")
    return get_public_attr(
        module,
        "SarsaConfig",
        predicate=lambda candidate: inspect.isclass(candidate)
        and candidate.__name__ == "SarsaConfig",
        description="SarsaConfig class",
    )


def make_sarsa_config(**overrides):
    config_class = require_sarsa_config_class()
    return call_with_supported_kwargs(config_class, **overrides)


def require_sarsa_training_entrypoint():
    module = import_public_module("rl_from_scratch.tabular.training")
    function = get_public_attr(
        module,
        "train_sarsa",
        predicate=callable,
        description="SARSA training entrypoint",
    )
    return module, function


def unwrap_result_field(result: Any, field_name: str):
    if isinstance(result, Mapping):
        return result.get(field_name)
    if hasattr(result, field_name):
        return getattr(result, field_name)
    return None


def maybe_set_config_value(config: Any, value: Any, *names: str) -> bool:
    for name in names:
        if hasattr(config, name):
            setattr(config, name, value)
            return True

    if isinstance(config, dict):
        lowered_to_actual = {key.lower(): key for key in config}
        for name in names:
            actual = lowered_to_actual.get(name.lower())
            if actual is not None:
                config[actual] = value
                return True

    return False


@pytest.fixture(autouse=True)
def _disable_greedy_video(monkeypatch):
    """Disable greedy video recording in all tests (slow and needs display).

    ``finalize_run`` in ``core.recording`` calls
    ``core.reporting.record_greedy_episode`` via module lookup, so patching
    the attribute on the module is sufficient for every training smoke test.
    """
    import rl_from_scratch.core.reporting as _rpt
    monkeypatch.setattr(_rpt, "record_greedy_episode", lambda *a, **kw: None)


@pytest.fixture
def small_config():
    config = make_config(
        bins=(3, 4, 5, 6),
        num_bins=(3, 4, 5, 6),
        state_bins=(3, 4, 5, 6),
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        num_episodes=3,
        episodes=3,
        total_episodes=3,
        render=False,
        seed=0,
    )
    maybe_set_config_value(config, (3, 4, 5, 6), "bins", "num_bins", "state_bins")
    maybe_set_config_value(
        config,
        3,
        "num_episodes",
        "episodes",
        "total_episodes",
        "number_of_epoch",
    )
    maybe_set_config_value(config, 0.5, "alpha", "ALPHA")
    maybe_set_config_value(config, 0.9, "gamma", "GAMMA")
    maybe_set_config_value(config, 0.0, "epsilon", "EPSILON")
    return config


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir
