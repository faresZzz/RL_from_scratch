from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.artifacts import save_checkpoint
from rl_from_scratch.core.base import BaseAgent


class DummyAgent(BaseAgent):
    def select_action(self, observation: Any, *, deterministic: bool = False) -> int:
        return 0

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        return {}

    def save(self, path: Path) -> Path:
        path.write_text(path.name, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "DummyAgent":
        return cls()


def checkpoint_names(checkpoint_dir: Path) -> set[str]:
    return {path.name for path in checkpoint_dir.iterdir()}


def test_save_checkpoint_prunes_by_numeric_step(tmp_path: Path) -> None:
    agent = DummyAgent()

    for step in (480_000, 490_000, 530_000, 1_000_000):
        save_checkpoint(agent, tmp_path, step=step, keep_last=3, keep_best=False)

    assert checkpoint_names(tmp_path) == {
        "checkpoint_490000.pt",
        "checkpoint_530000.pt",
        "checkpoint_1000000.pt",
    }


def test_save_checkpoint_never_prunes_best_checkpoint(tmp_path: Path) -> None:
    agent = DummyAgent()
    best_reward: dict[str, float] = {}

    for step, reward in (
        (480_000, 1.0),
        (490_000, 2.0),
        (530_000, 3.0),
        (1_000_000, 4.0),
    ):
        save_checkpoint(
            agent,
            tmp_path,
            step=step,
            keep_last=3,
            current_reward=reward,
            _best_reward=best_reward,
        )

    assert "best.pt" in checkpoint_names(tmp_path)
    assert (tmp_path / "best.pt").read_text(encoding="utf-8") == "checkpoint_1000000.pt"
