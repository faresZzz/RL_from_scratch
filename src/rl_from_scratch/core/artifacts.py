"""Utilities for reproducible experiment artifacts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
import json
from pathlib import Path
from time import strftime
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from rl_from_scratch.core.base import BaseAgent


@dataclass(frozen=True)
class ExperimentPaths:
    run_dir: Path
    figure_dir: Path
    checkpoint_dir: Path
    history_path: Path
    metrics_path: Path
    config_path: Path


def _unique_run_dir(run_dir: Path) -> Path:
    candidate = run_dir
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = run_dir.with_name(f"{run_dir.name}-{suffix:02d}")
    return candidate


def create_experiment_run(
    root: str | Path,
    *,
    approach: str,
    run_name: str | None = None,
    config: Mapping[str, object] | None = None,
) -> ExperimentPaths:
    run_id = run_name or strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(Path(root) / approach / run_id)
    figure_dir = run_dir / "figures"
    checkpoint_dir = run_dir / "checkpoints"
    figure_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    paths = ExperimentPaths(
        run_dir=run_dir,
        figure_dir=figure_dir,
        checkpoint_dir=checkpoint_dir,
        history_path=run_dir / "history.json",
        metrics_path=run_dir / "metrics.json",
        config_path=run_dir / "config.json",
    )
    if config is not None:
        save_json(dict(config), paths.config_path)
    return paths


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix("checkpoint_"))


def save_checkpoint(
    agent: BaseAgent,
    checkpoint_dir: str | Path,
    *,
    step: int,
    keep_last: int = 3,
    keep_best: bool = True,
    current_reward: float | None = None,
    _best_reward: dict[str, float] | None = None,
) -> Path:
    """Save an agent checkpoint, prune old ones, optionally track best.

    Parameters
    ----------
    agent:
        Any agent implementing ``BaseAgent.save(path)``.
    checkpoint_dir:
        Directory where checkpoint files are written.
    step:
        Current step (episode or timestep) used in the filename.
    keep_last:
        Number of most-recent checkpoints to retain; older ones are deleted.
    keep_best:
        Whether to maintain a ``best.pt`` copy when *current_reward* improves.
    current_reward:
        Reward used to determine if this checkpoint is the best so far.
        Ignored when *keep_best* is ``False``.
    _best_reward:
        Internal mutable state dict (``{"value": float}``) that tracks the
        running best reward across calls.  Callers should **not** set this
        directly — instead, create one dict and pass the same reference on
        every call so the function can update it in-place.

    Returns
    -------
    Path
        The path to the newly written checkpoint file.
    """
    if keep_last <= 0:
        raise ValueError("keep_last must be positive.")

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = root / f"checkpoint_{step:06d}.pt"
    agent.save(checkpoint_path)

    # Prune: keep only the *keep_last* most-recent numbered checkpoints.
    numbered = sorted(root.glob("checkpoint_*.pt"), key=_checkpoint_step)
    for stale in numbered[:-keep_last]:
        stale.unlink()

    # Best tracking
    if keep_best and current_reward is not None and _best_reward is not None:
        if current_reward > _best_reward.get("value", float("-inf")):
            _best_reward["value"] = current_reward
            best_path = root / "best.pt"
            shutil.copy2(checkpoint_path, best_path)

    return checkpoint_path


def save_json(payload: object, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
