"""Lazy PettingZoo Connect-4 adapter for MuZero demos/tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_from_scratch.muzero.replay import GameHistory


def _make_env(render_mode: str | None = None):
    from pettingzoo.classic import connect_four_v3

    return connect_four_v3.env(render_mode=render_mode)


def _flatten_observation(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    board = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
    mask = np.asarray(observation["action_mask"], dtype=np.int8)
    return board, mask


class ConnectFourAdapter:
    """Thin AEC adapter exposing flat observations and legal-action masks."""

    def __init__(self) -> None:
        self.env = _make_env()
        self.current_agent: str | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.env.reset(seed=seed)
        self.current_agent = self.env.agent_selection
        observation, _, terminated, truncated, _ = self.env.last()
        if terminated or truncated:
            raise RuntimeError("ConnectFour reset returned a finished state.")
        return _flatten_observation(observation)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], np.ndarray, int]:
        acting_agent = self.current_agent
        if acting_agent is None:
            raise RuntimeError("ConnectFourAdapter.step called before reset.")
        self.env.step(action)
        actor_reward = float(self.env.rewards.get(acting_agent, 0.0))
        self.current_agent = self.env.agent_selection
        observation, reward, terminated, truncated, info = self.env.last()
        if terminated or truncated:
            board = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
            mask = np.asarray(observation["action_mask"], dtype=np.int8)
            return board, actor_reward, bool(terminated), bool(truncated), info, mask, -1
        board, mask = _flatten_observation(observation)
        to_play = 1 if self.current_agent == "player_0" else -1
        return board, actor_reward, False, False, info, mask, to_play

    def close(self) -> None:
        self.env.close()


def self_play_connect_four(
    agent: Any,
    *,
    max_moves: int = 42,
    seed: int = 0,
) -> GameHistory:
    adapter = ConnectFourAdapter()
    observation, legal_actions = adapter.reset(seed=seed)
    to_play = 1
    game = GameHistory(observations=[observation.copy()])
    try:
        for _ in range(max_moves):
            action = agent.select_action(
                observation,
                legal_actions=legal_actions,
                to_play=to_play,
            )
            next_obs, reward, terminated, truncated, _info, next_mask, next_to_play = adapter.step(action)
            game.actions.append(int(action))
            game.rewards.append(float(reward))
            game.root_values.append(float(agent._pending_root_value))
            game.child_visits.append(agent._pending_child_visits.copy())
            game.to_play.append(int(to_play))
            game.observations.append(next_obs.copy())
            observation = next_obs
            legal_actions = next_mask
            to_play = next_to_play
            if terminated or truncated:
                break
    finally:
        adapter.close()
    return game


def evaluate_vs_random(
    agent: Any,
    *,
    num_games: int = 3,
    max_moves: int = 42,
    seed: int = 0,
) -> dict[str, float]:
    rewards: list[float] = []
    wins = 0
    for game_index in range(num_games):
        adapter = ConnectFourAdapter()
        observation, legal_actions = adapter.reset(seed=seed + game_index)
        to_play = 1
        total_reward = 0.0
        try:
            for _ in range(max_moves):
                acting_to_play = to_play
                if acting_to_play == 1:
                    action = agent.select_action(
                        observation,
                        deterministic=True,
                        legal_actions=legal_actions,
                        to_play=acting_to_play,
                    )
                else:
                    legal_indices = np.flatnonzero(legal_actions)
                    action = int(legal_indices[0])
                observation, reward, terminated, truncated, _info, legal_actions, to_play = adapter.step(action)
                # Keep the score from player_0's perspective even when the
                # random opponent is the actor that receives the terminal reward.
                total_reward += float(reward) if acting_to_play == 1 else -float(reward)
                if terminated or truncated:
                    break
        finally:
            adapter.close()
        rewards.append(total_reward)
        if total_reward > 0.0:
            wins += 1
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "win_rate": float(wins / max(1, num_games)),
    }
