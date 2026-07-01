"""World models used by tabular and deep dyna agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelTransition:
    """One stored model transition."""

    state: tuple[int, ...]
    action: int
    reward: float
    next_state: tuple[int, ...]
    done: bool
    last_seen: int


class TabularWorldModel:
    """Deterministic one-step tabular world model."""

    def __init__(self) -> None:
        self._transitions: dict[tuple[tuple[int, ...], int], ModelTransition] = {}

    @property
    def seen_keys(self) -> set[tuple[tuple[int, ...], int]]:
        return set(self._transitions)

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
        *,
        time_step: int = 0,
    ) -> None:
        key = (tuple(state), int(action))
        self._transitions[key] = ModelTransition(
            state=tuple(state),
            action=int(action),
            reward=float(reward),
            next_state=tuple(next_state),
            done=bool(done),
            last_seen=int(time_step),
        )

    def ensure_all_actions_for_state(
        self,
        state: tuple[int, ...],
        action_count: int,
        *,
        time_step: int = 0,
    ) -> None:
        state = tuple(state)
        for action in range(action_count):
            key = (state, action)
            if key in self._transitions:
                continue
            self._transitions[key] = ModelTransition(
                state=state,
                action=action,
                reward=0.0,
                next_state=state,
                done=False,
                last_seen=int(time_step),
            )

    def sample(self, rng: np.random.Generator) -> ModelTransition | None:
        if not self._transitions:
            return None
        keys = tuple(self._transitions)
        index = int(rng.integers(len(keys)))
        return self._transitions[keys[index]]

    def state_dict(self) -> dict[str, Any]:
        return {
            "transitions": [
                {
                    "state": transition.state,
                    "action": transition.action,
                    "reward": transition.reward,
                    "next_state": transition.next_state,
                    "done": transition.done,
                    "last_seen": transition.last_seen,
                }
                for transition in self._transitions.values()
            ]
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._transitions.clear()
        for item in payload.get("transitions", []):
            self.update(
                tuple(item["state"]),
                int(item["action"]),
                float(item["reward"]),
                tuple(item["next_state"]),
                bool(item["done"]),
                time_step=int(item.get("last_seen", 0)),
            )

    def __len__(self) -> int:
        return len(self._transitions)
