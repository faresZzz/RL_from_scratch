"""Tests for config registry from rl_from_scratch.core.config."""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

import rl_from_scratch  # noqa: F401  # trigger config registration
from rl_from_scratch.core.base import BaseConfig
from rl_from_scratch.core.config import (
    AGENT_FACTORIES,
    CONFIG_REGISTRY,
    load_config,
    register_agent,
    register_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_register_config_adds_to_registry():
    name = "_test_reg_config"
    # Clean up in case of re-runs
    CONFIG_REGISTRY.pop(name, None)

    @register_config(name)
    @dataclass
    class _RegTestConfig(BaseConfig):
        approach: str = name

    assert name in CONFIG_REGISTRY
    assert CONFIG_REGISTRY[name] is _RegTestConfig

    # Cleanup
    CONFIG_REGISTRY.pop(name, None)


def test_load_config_dispatches_by_approach(tmp_path):
    name = "_dispatch_test"
    CONFIG_REGISTRY.pop(name, None)

    @register_config(name)
    @dataclass
    class _DispatchConfig(BaseConfig):
        approach: str = name
        lr: float = 0.01

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
            approach: {name}
            lr: 0.001
        """)
    )

    config = load_config(str(yaml_path))
    assert isinstance(config, _DispatchConfig)
    assert config.approach == name
    assert config.lr == pytest.approx(0.001)

    # Cleanup
    CONFIG_REGISTRY.pop(name, None)


def test_load_config_rejects_unknown_keys(tmp_path):
    name = "_strict_load_test"
    CONFIG_REGISTRY.pop(name, None)

    @register_config(name)
    @dataclass
    class _StrictLoadConfig(BaseConfig):
        approach: str = name
        lr: float = 0.01

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
            approach: {name}
            lr: 0.001
            lrr: 0.1
        """)
    )

    with pytest.raises(ValueError, match="Unknown config keys.*lrr"):
        load_config(str(yaml_path))

    CONFIG_REGISTRY.pop(name, None)


def test_config_from_dict_rejects_unknown_keys():
    name = "_strict_from_dict_test"
    CONFIG_REGISTRY.pop(name, None)

    @register_config(name)
    @dataclass
    class _StrictFromDictConfig(BaseConfig):
        approach: str = name
        lr: float = 0.01

    with pytest.raises(ValueError, match="Unknown config keys.*lrr"):
        _StrictFromDictConfig.from_dict({"approach": name, "lr": 0.001, "lrr": 0.1})

    CONFIG_REGISTRY.pop(name, None)


def test_all_committed_yaml_configs_load_strictly():
    config_paths = sorted((PROJECT_ROOT / "configs").glob("*/*.yaml"))
    assert config_paths, "Expected at least one YAML config."

    for config_path in config_paths:
        load_config(config_path)


def test_register_agent_adds_factory():
    name = "_test_reg_agent"
    AGENT_FACTORIES.pop(name, None)

    @register_agent(name)
    def _make_test_agent(config):
        return None

    assert name in AGENT_FACTORIES
    assert AGENT_FACTORIES[name] is _make_test_agent

    # Cleanup
    AGENT_FACTORIES.pop(name, None)
