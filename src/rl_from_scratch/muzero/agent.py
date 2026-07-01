"""Autonomous MuZero agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device
from rl_from_scratch.muzero.mcts import Node, run_mcts
from rl_from_scratch.muzero.networks import (
    Dynamics,
    Prediction,
    Representation,
    scalar_to_support,
    support_to_scalar,
)
from rl_from_scratch.muzero.replay import GameHistory, ReplayBuffer, make_target


def _soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    if weights is None:
        return per_sample.mean()
    weights = weights.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


class MuZeroAgent(BaseAgent):
    """Pedagogical MuZero agent with latent planning and replay."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        *,
        encoding_dim: int = 8,
        hidden_dim: int = 32,
        support_size: int = 10,
        replay_capacity: int = 500,
        batch_size: int = 64,
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        lr: float = 0.02,
        weight_decay: float = 1e-4,
        grad_clip: float = 10.0,
        value_loss_weight: float = 0.25,
        discount: float = 0.997,
        num_simulations: int = 25,
        pb_c_base: float = 19_652.0,
        pb_c_init: float = 1.25,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25,
        root_temperature: float = 1.0,
        root_temperature_drop_episode: int | None = None,
        two_player: bool = False,
        seed: int = 0,
        device: str = "auto",
        **_kwargs: Any,
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.support_size = support_size
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.value_loss_weight = value_loss_weight
        self.discount = discount
        self.num_simulations = num_simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.root_temperature = root_temperature
        self.root_temperature_drop_episode = root_temperature_drop_episode
        self.two_player = two_player
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.representation = Representation(obs_dim, hidden_dim, encoding_dim).to(self.device)
        self.dynamics = Dynamics(encoding_dim, num_actions, hidden_dim, support_size).to(self.device)
        self.prediction = Prediction(encoding_dim, hidden_dim, num_actions, support_size).to(self.device)
        params = (
            list(self.representation.parameters())
            + list(self.dynamics.parameters())
            + list(self.prediction.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        self.replay_buffer = ReplayBuffer(replay_capacity, seed=seed)
        self.current_game = GameHistory()
        self._last_next_observation: np.ndarray | None = None
        self._pending_root_value = 0.0
        self._pending_child_visits = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)
        self._pending_to_play = 1
        self._episode_index = 0

    def _model_fns(self) -> dict[str, Any]:
        def prediction(hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            policy_logits, value_logits = self.prediction(hidden_state)
            value = support_to_scalar(
                torch.softmax(value_logits, dim=-1),
                self.support_size,
                apply_inverse=True,
            )
            return policy_logits, value

        def dynamics(hidden_state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            next_hidden, reward_logits = self.dynamics(hidden_state, actions)
            reward = support_to_scalar(
                torch.softmax(reward_logits, dim=-1),
                self.support_size,
                apply_inverse=True,
            )
            return next_hidden, reward

        return {"prediction": prediction, "dynamics": dynamics}

    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool = False,
        legal_actions: np.ndarray | None = None,
        to_play: int = 1,
    ) -> int:
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            hidden_state = self.representation(obs_tensor)
            root = Node(prior=1.0, to_play=to_play, hidden_state=hidden_state)
            run_mcts(
                root,
                self._model_fns(),
                self,
                legal_actions=legal_actions,
                to_play=to_play,
                rng=self.rng,
                add_root_noise=not deterministic,
            )

        visit_counts = np.zeros(self.num_actions, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        if visit_counts.sum() <= 0.0:
            if legal_actions is None:
                visit_counts[:] = 1.0
            else:
                visit_counts = np.asarray(legal_actions, dtype=np.float32)

        temperature = self.root_temperature
        if deterministic or (
            self.root_temperature_drop_episode is not None
            and self._episode_index >= self.root_temperature_drop_episode
        ):
            temperature = 1e-8

        legal_mask = None if legal_actions is None else np.asarray(legal_actions, dtype=np.float32)
        if temperature <= 1e-6:
            masked = visit_counts.copy()
            if legal_mask is not None:
                masked *= legal_mask
            action = int(masked.argmax())
        else:
            probs = visit_counts ** (1.0 / temperature)
            if legal_mask is not None:
                probs *= legal_mask
            total = float(probs.sum())
            if total <= 0.0:
                probs = legal_mask if legal_mask is not None else np.ones(self.num_actions, dtype=np.float32)
                total = float(probs.sum())
            probs = probs / total
            action = int(self.rng.choice(self.num_actions, p=probs))

        self._pending_root_value = root.value()
        self._pending_child_visits = (
            visit_counts / max(1e-8, visit_counts.sum())
        ).astype(np.float32)
        self._pending_to_play = to_play
        return action

    def snapshot_acting_state(self) -> dict[str, Any]:
        """Save transient acting/search state so evaluation can be side-effect free."""
        return {
            "pending_root_value": self._pending_root_value,
            "pending_child_visits": self._pending_child_visits.copy(),
            "pending_to_play": self._pending_to_play,
            "episode_index": self._episode_index,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_acting_state(self, state: dict[str, Any]) -> None:
        """Restore transient acting/search state saved by ``snapshot_acting_state``."""
        self._pending_root_value = float(state["pending_root_value"])
        self._pending_child_visits = np.asarray(
            state["pending_child_visits"],
            dtype=np.float32,
        )
        self._pending_to_play = int(state["pending_to_play"])
        self._episode_index = int(state["episode_index"])
        self.rng.bit_generator.state = state["rng_state"]

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
        *,
        to_play: int | None = None,
    ) -> None:
        self.current_game.observations.append(np.asarray(obs, dtype=np.float32))
        self.current_game.actions.append(int(action))
        self.current_game.rewards.append(float(reward))
        self.current_game.root_values.append(float(self._pending_root_value))
        self.current_game.child_visits.append(self._pending_child_visits.copy())
        self.current_game.to_play.append(int(self._pending_to_play if to_play is None else to_play))
        self._last_next_observation = np.asarray(next_obs, dtype=np.float32)
        if done:
            self.episode_ended()

    def episode_ended(self) -> None:
        if self._last_next_observation is not None:
            self.current_game.observations.append(self._last_next_observation.copy())
        if self.current_game.actions:
            self.replay_buffer.add_game(self.current_game)
            self._episode_index += 1
        self.current_game = GameHistory()
        self._last_next_observation = None
        self._pending_root_value = 0.0
        self._pending_child_visits = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        self._pending_to_play = 1

    def learn_step(self, **_: Any) -> dict[str, float]:
        if not self.replay_buffer.games or self.replay_buffer.num_positions < self.batch_size:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "reward_loss": 0.0,
                "root_value_mean": 0.0,
            }

        positions = self.replay_buffer.sample_positions(self.batch_size)
        batches = [
            make_target(self.replay_buffer.games[game_index], position, self)
            for game_index, position in positions
        ]
        observations = torch.as_tensor(
            np.stack([batch[0]["observation"] for batch in batches]),
            dtype=torch.float32,
            device=self.device,
        )
        hidden_state = self.representation(observations)

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)
        root_values: list[float] = []

        for step in range(self.num_unroll_steps + 1):
            step_targets = [batch[step] for batch in batches]
            policy_targets = torch.as_tensor(
                np.stack([target["policy"] for target in step_targets]),
                dtype=torch.float32,
                device=self.device,
            )
            policy_masks = torch.as_tensor(
                [target["policy_mask"] for target in step_targets],
                dtype=torch.float32,
                device=self.device,
            )
            value_targets = torch.as_tensor(
                [target["value"] for target in step_targets],
                dtype=torch.float32,
                device=self.device,
            )
            policy_logits, value_logits = self.prediction(hidden_state)
            total_policy_loss = total_policy_loss + _soft_cross_entropy(
                policy_logits,
                policy_targets,
                weights=policy_masks,
            )
            total_value_loss = total_value_loss + _soft_cross_entropy(
                value_logits,
                scalar_to_support(value_targets, self.support_size, apply_transform=True),
            )
            if step == 0:
                with torch.no_grad():
                    decoded = support_to_scalar(
                        torch.softmax(value_logits, dim=-1),
                        self.support_size,
                        apply_inverse=True,
                    )
                    root_values.extend(decoded.detach().cpu().tolist())
            if step == self.num_unroll_steps:
                continue
            action_targets = torch.as_tensor(
                [target["action"] for target in step_targets],
                dtype=torch.long,
                device=self.device,
            )
            reward_targets = torch.as_tensor(
                [target["reward"] for target in step_targets],
                dtype=torch.float32,
                device=self.device,
            )
            hidden_state, reward_logits = self.dynamics(hidden_state, action_targets)
            total_reward_loss = total_reward_loss + _soft_cross_entropy(
                reward_logits,
                scalar_to_support(reward_targets, self.support_size, apply_transform=True),
            )

        denom = float(self.num_unroll_steps + 1)
        policy_loss = total_policy_loss / denom
        value_loss = total_value_loss / denom
        reward_loss = total_reward_loss / max(1.0, float(self.num_unroll_steps))
        loss = policy_loss + self.value_loss_weight * value_loss + reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.representation.parameters())
            + list(self.dynamics.parameters())
            + list(self.prediction.parameters()),
            self.grad_clip,
        )
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "root_value_mean": float(np.mean(root_values)) if root_values else 0.0,
        }

    def save(self, path: Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "init_kwargs": {
                    "obs_dim": self.obs_dim,
                    "num_actions": self.num_actions,
                    "encoding_dim": self.encoding_dim,
                    "hidden_dim": self.hidden_dim,
                    "support_size": self.support_size,
                    "replay_capacity": self.replay_capacity,
                    "batch_size": self.batch_size,
                    "num_unroll_steps": self.num_unroll_steps,
                    "td_steps": self.td_steps,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "grad_clip": self.grad_clip,
                    "value_loss_weight": self.value_loss_weight,
                    "discount": self.discount,
                    "num_simulations": self.num_simulations,
                    "pb_c_base": self.pb_c_base,
                    "pb_c_init": self.pb_c_init,
                    "dirichlet_alpha": self.dirichlet_alpha,
                    "exploration_fraction": self.exploration_fraction,
                    "root_temperature": self.root_temperature,
                    "root_temperature_drop_episode": self.root_temperature_drop_episode,
                    "two_player": self.two_player,
                    "seed": self.seed,
                    "device": str(self.device),
                },
                "representation": self.representation.state_dict(),
                "dynamics": self.dynamics.state_dict(),
                "prediction": self.prediction.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "replay_buffer": self.replay_buffer.state_dict(),
                "episode_index": self._episode_index,
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "MuZeroAgent":
        payload = torch.load(Path(path), weights_only=False)
        init_kwargs = dict(payload["init_kwargs"])
        init_kwargs.update(kwargs)
        agent = cls(**init_kwargs)
        agent.representation.load_state_dict(payload["representation"])
        agent.dynamics.load_state_dict(payload["dynamics"])
        agent.prediction.load_state_dict(payload["prediction"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.replay_buffer.load_state_dict(payload.get("replay_buffer", {}))
        agent._episode_index = int(payload.get("episode_index", 0))
        return agent
