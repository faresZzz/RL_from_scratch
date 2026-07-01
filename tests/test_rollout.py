"""Regression tests for local experience collection living in training.py."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src/rl_from_scratch"
ALGORITHM_PACKAGES = (
    "actor_critic",
    "deep_q",
    "deterministic_actor_critic",
    "reinforce",
    "sac",
    "tabular",
    "trust_region",
)


def _top_level_function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def test_algorithm_packages_keep_experience_collection_in_training_modules() -> None:
    """Each algorithm package must have a training.py with at least one train_xxx entry-point.

    Packages that inline the episode loop (SAC, deterministic_actor_critic) are not
    required to expose ``train_one_episode``.  Packages that delegate evaluation to
    ``core.evaluate`` are not required to re-export ``evaluate``.
    """
    for package in ALGORITHM_PACKAGES:
        package_dir = PACKAGE_ROOT / package
        training_path = package_dir / "training.py"
        rollout_path = package_dir / "rollout.py"

        assert training_path.exists(), f"Missing training module for {package}"
        assert not rollout_path.exists(), f"Legacy rollout module should not exist for {package}"

        function_names = _top_level_function_names(training_path)

        # Every training module must have at least one public train_xxx entry point.
        train_fns = [n for n in function_names if n.startswith("train_") and not n.startswith("_")]
        assert train_fns, (
            f"{package}/training.py must expose at least one public train_xxx function, "
            f"got: {sorted(function_names)}"
        )
