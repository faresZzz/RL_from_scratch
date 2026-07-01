"""Replay storage and MuZero target construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GameHistory:
    """One self-play trajectory with search statistics."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)
    child_visits: list[np.ndarray] = field(default_factory=list)
    to_play: list[int] = field(default_factory=list)

    def clone(self) -> "GameHistory":
        return GameHistory(
            observations=[obs.copy() for obs in self.observations],
            actions=list(self.actions),
            rewards=list(self.rewards),
            root_values=list(self.root_values),
            child_visits=[visits.copy() for visits in self.child_visits],
            to_play=list(self.to_play),
        )

    def __len__(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    """FIFO buffer of completed self-play games."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.games: list[GameHistory] = []
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.games)

    @property
    def num_positions(self) -> int:
        return sum(max(1, len(game.observations) - 1) for game in self.games)

    def add_game(self, game: GameHistory) -> None:
        self.games.append(game.clone())
        if len(self.games) > self.capacity:
            self.games.pop(0)

    def sample_positions(self, batch_size: int) -> list[tuple[int, int]]:
        if not self.games:
            raise RuntimeError("ReplayBuffer is empty.")
        lengths = np.asarray(
            [max(1, len(game.observations) - 1) for game in self.games],
            dtype=np.float64,
        )
        probabilities = lengths / lengths.sum()
        positions: list[tuple[int, int]] = []
        for _ in range(batch_size):
            game_index = int(self.rng.choice(len(self.games), p=probabilities))
            game = self.games[game_index]
            position = int(self.rng.integers(int(lengths[game_index])))
            positions.append((game_index, position))
        return positions

    def state_dict(self) -> dict[str, Any]:
        serialized_games = []
        for game in self.games:
            serialized_games.append(
                {
                    "observations": [obs.tolist() for obs in game.observations],
                    "actions": list(game.actions),
                    "rewards": list(game.rewards),
                    "root_values": list(game.root_values),
                    "child_visits": [visits.tolist() for visits in game.child_visits],
                    "to_play": list(game.to_play),
                }
            )
        return {
            "capacity": self.capacity,
            "games": serialized_games,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.capacity = int(payload.get("capacity", self.capacity))
        self.games = []
        for game_data in payload.get("games", []):
            self.games.append(
                GameHistory(
                    observations=[
                        np.asarray(obs, dtype=np.float32)
                        for obs in game_data.get("observations", [])
                    ],
                    actions=[int(action) for action in game_data.get("actions", [])],
                    rewards=[float(reward) for reward in game_data.get("rewards", [])],
                    root_values=[
                        float(value) for value in game_data.get("root_values", [])
                    ],
                    child_visits=[
                        np.asarray(visits, dtype=np.float32)
                        for visits in game_data.get("child_visits", [])
                    ],
                    to_play=[int(player) for player in game_data.get("to_play", [])],
                )
            )
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state


def make_target(game: GameHistory, position: int, config: Any) -> list[dict[str, Any]]:
    """Construct MuZero targets aligned on (obs_t, action_t, reward_{t+1})."""

    num_unroll_steps = int(getattr(config, "num_unroll_steps", 5))
    td_steps = int(getattr(config, "td_steps", 10))
    discount = float(getattr(config, "discount", 0.997))
    num_actions = len(game.child_visits[0]) if game.child_visits else 1
    zero_policy = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)
    last_observation = game.observations[-1]

    targets: list[dict[str, Any]] = []
    for current_index in range(position, position + num_unroll_steps + 1):
        observation = (
            game.observations[current_index]
            if current_index < len(game.observations)
            else last_observation
        )
        if current_index < len(game.actions):
            action = int(game.actions[current_index])
            reward = float(game.rewards[current_index])
            policy = game.child_visits[current_index].astype(np.float32, copy=True)
            policy_mask = 1.0
        else:
            action = 0
            reward = 0.0
            policy = zero_policy.copy()
            policy_mask = 0.0

        # Value target in the perspective of the player to move at current_index
        # (negamax): an opponent step's reward/bootstrap enters with a flipped
        # sign.  Single-player envs leave to_play constant (or empty), so every
        # sign is +1 and this reduces to the plain discounted return.
        def _player(idx: int) -> int:
            return game.to_play[idx] if idx < len(game.to_play) else 1

        value = 0.0
        if current_index < len(game.rewards):
            perspective = _player(current_index)
            for step in range(td_steps):
                reward_index = current_index + step
                if reward_index >= len(game.rewards):
                    break
                sign = 1.0 if _player(reward_index) == perspective else -1.0
                value += sign * (discount**step) * float(game.rewards[reward_index])
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(game.root_values):
                sign = 1.0 if _player(bootstrap_index) == perspective else -1.0
                value += sign * (discount**td_steps) * float(game.root_values[bootstrap_index])

        targets.append(
            {
                "observation": observation.astype(np.float32, copy=True),
                "action": action,
                "reward": reward,
                "value": float(value),
                "policy": policy,
                "policy_mask": policy_mask,
                "to_play": game.to_play[current_index] if current_index < len(game.to_play) else 1,
            }
        )
    return targets
