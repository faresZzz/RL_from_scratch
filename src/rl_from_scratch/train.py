"""Command-line entry point for reinforcement learning experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

# Side-effect import: importing rl_from_scratch triggers auto-discovery
# of all sub-packages, which registers all @register_config/@register_agent decorators.
import rl_from_scratch  # noqa: F401
from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train RL agents from scratch.")
    parser.add_argument("--config", type=Path, required=False, help="Path to a YAML configuration file.")
    parser.add_argument("--approach", default=None, choices=sorted(AGENT_FACTORIES))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--render", action="store_true")
    return parser


def merge_cli_overrides(config: object, args: argparse.Namespace) -> object:
    """Apply CLI overrides to a config that supports to_dict/from_dict."""
    payload = config.to_dict()  # type: ignore[union-attr]
    for field_name in ("approach", "episodes", "seed", "run_name"):
        value = getattr(args, field_name)
        if value is not None:
            payload[field_name] = value
    if args.output_dir is not None:
        payload["output_dir"] = args.output_dir
    if args.render:
        payload["render"] = True
    config_cls = type(config)
    return config_cls.from_dict(payload)


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for the training CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load or create the configuration
    if args.config:
        config = load_config(args.config)
    else:
        approach = args.approach # need to be set if no config file provided
        config_cls = CONFIG_REGISTRY.get(approach)
        if config_cls is None:
            raise ValueError(f"Unknown approach: {approach}")
        config = config_cls()

    config = merge_cli_overrides(config, args)

    # Retrieve the registered factory
    approach = config.approach  # type: ignore[union-attr]
    factory = AGENT_FACTORIES.get(approach)
    if factory is None:
        registered = ", ".join(sorted(AGENT_FACTORIES)) or "(none)"
        raise ValueError(
            f"No factory registered for approach '{approach}'. "
            f"Registered: {registered}."
        )

    result = factory(
        config,
        output_dir=args.output_dir or config.output_dir,  # type: ignore[union-attr]
        run_name=args.run_name or config.run_name,  # type: ignore[union-attr]
        seed=args.seed,
        render=args.render if args.render else None,
    )

    print(f"Run directory: {result['paths'].run_dir}")
    print(f"Mean reward: {result['metrics']['mean_reward']:.2f}")
    print(f"Best reward: {result['metrics']['best_reward']:.2f}")


if __name__ == "__main__":
    main()
