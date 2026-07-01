"""Core config registry and YAML helpers."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any, Callable

import yaml

from rl_from_scratch.core.base import BaseConfig

CONFIG_REGISTRY: dict[str, type[BaseConfig]] = {}
AGENT_FACTORIES: dict[str, Callable[..., Any]] = {}


def strict_dataclass_from_dict(
    cls: type[BaseConfig],
    payload: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
    ignored_keys: set[str] | None = None,
    converters: dict[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Construct a config dataclass and reject silent YAML/key drift.

    Config files are part of the experiment protocol.  A misspelled key should
    fail loudly instead of being filtered out and making the run look configured
    while using a default value.
    """
    data = dict(payload)
    for key in ignored_keys or set():
        data.pop(key, None)

    for source, target in (aliases or {}).items():
        if source not in data:
            continue
        if target in data:
            raise ValueError(
                f"Ambiguous config keys for {cls.__name__}: both "
                f"'{source}' and '{target}' were provided."
            )
        data[target] = data.pop(source)

    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"Unknown config keys for {cls.__name__}: {keys}")

    for key, converter in (converters or {}).items():
        if key in data:
            data[key] = converter(data[key])

    return cls(**data)


def register_config(approach: str) -> Callable[[type[BaseConfig]], type[BaseConfig]]:
    """Register a config dataclass under an approach name."""

    def _decorator(cls: type[BaseConfig]) -> type[BaseConfig]:
        if approach in CONFIG_REGISTRY:
            raise ValueError(
                f"Approach '{approach}' already registered "
                f"to {CONFIG_REGISTRY[approach].__name__}.",
            )
        CONFIG_REGISTRY[approach] = cls
        return cls

    return _decorator


def register_agent(approach: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a training function or factory under an approach name."""

    def _decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if approach in AGENT_FACTORIES:
            raise ValueError(
                f"Approach '{approach}' already registered "
                f"to {AGENT_FACTORIES[approach]!r}.",
            )
        AGENT_FACTORIES[approach] = factory
        return factory

    return _decorator


def load_config(path: str | Path) -> BaseConfig:
    """Load a YAML config and dispatch from the registered ``approach``."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}.")

    approach = data.get("approach")
    if approach is None:
        raise ValueError(f"Missing 'approach' key in {config_path}.")

    config_cls = CONFIG_REGISTRY.get(approach)
    if config_cls is None:
        registered = ", ".join(sorted(CONFIG_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown approach '{approach}'. Registered: {registered}.",
        )

    # Each config's ``from_dict`` already calls ``strict_dataclass_from_dict``
    # (with its own aliases/ignored_keys/converters), so it is the single source
    # of strict key validation — a redundant pre-check here would wrongly reject
    # legitimately aliased/ignored YAML keys.
    return config_cls.from_dict(data)


def apply_overrides(
    config: BaseConfig,
    config_cls: type[BaseConfig],
    *,
    seed: int | None = None,
    run_name: str | None = None,
    output_dir: str | None = None,
) -> BaseConfig:
    """Re-create *config* with CLI/caller overrides applied.

    This pattern (``to_dict`` → patch → ``from_dict``) is used by every
    ``train_xxx()`` entry-point to let callers override seed, run_name, or
    output_dir without mutating the original config.
    """
    overrides = config.to_dict()
    if seed is not None:
        overrides["seed"] = seed
    if run_name is not None:
        overrides["run_name"] = run_name
    if output_dir is not None:
        overrides["output_dir"] = str(output_dir)
    return config_cls.from_dict(overrides)


def build_agent(agent_cls: type, config: BaseConfig, **env_kwargs: Any) -> Any:
    """Build an agent from config fields plus environment-specific kwargs."""
    sig = inspect.signature(agent_cls.__init__)
    config_dict = dataclasses.asdict(config)

    agent_kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in env_kwargs:
            continue
        if param.kind == param.VAR_KEYWORD:
            continue
        if name in config_dict:
            agent_kwargs[name] = config_dict[name]

    return agent_cls(**env_kwargs, **agent_kwargs)
