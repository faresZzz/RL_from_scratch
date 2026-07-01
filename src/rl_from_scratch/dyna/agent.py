"""Autonomous agents for Dyna-Q, Dyna-Q+, and Deep Dyna."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.utils import resolve_device
from rl_from_scratch.dyna.buffer import ReplayBuffer
from rl_from_scratch.dyna.model import TabularWorldModel
from rl_from_scratch.dyna.network import NeuralDynamicsModel, QNetwork


class DynaQAgent(BaseAgent):
    """Tabular Dyna-Q agent with an explicit world model."""

    def __init__(
        self,
        state_shape: tuple[int, ...],
        action_count: int,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        planning_steps: int = 10,
        rng: np.random.Generator | None = None,
        q_table: torch.Tensor | None = None,
    ) -> None:
        self.state_shape = state_shape
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.planning_steps = planning_steps
        self.rng = rng or np.random.default_rng()
        self.world_model = TabularWorldModel()
        self.total_updates = 0

        expected_shape = (*state_shape, action_count)
        self.q_table = (
            torch.zeros(expected_shape, dtype=torch.float32)
            if q_table is None
            else q_table.to(dtype=torch.float32).clone()
        )

    def select_action(self, observation: Any, *, deterministic: bool = False) -> int:
        state = tuple(observation)
        if not deterministic and float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(self.action_count))
        values = self.q_table[state]
        max_value = torch.max(values)
        best_actions = torch.nonzero(values == max_value, as_tuple=False).flatten()
        index = int(self.rng.integers(len(best_actions)))
        return int(best_actions[index].item())

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        return self.learn_real_transition(
            state=kwargs["state"],
            action=kwargs["action"],
            reward=kwargs["reward"],
            next_state=kwargs["next_state"],
            done=kwargs["done"],
        )

    def learn_real_transition(
        self,
        *,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> dict[str, float]:
        td_error = self._q_update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.world_model.update(
            state,
            action,
            reward,
            next_state,
            done,
            time_step=self.total_updates,
        )
        self.total_updates += 1
        return {
            "real_td_error": abs(td_error),
            "planning_td_error": 0.0,
            "exploration_bonus": 0.0,
            "model_buffer_size": float(len(self.world_model)),
        }

    def planning_step(self) -> dict[str, float]:
        transition = self.world_model.sample(self.rng)
        if transition is None:
            return {
                "real_td_error": 0.0,
                "planning_td_error": 0.0,
                "exploration_bonus": 0.0,
                "model_buffer_size": float(len(self.world_model)),
            }

        bonus = self._planning_bonus(transition.last_seen)
        td_error = self._q_update(
            state=transition.state,
            action=transition.action,
            reward=transition.reward + bonus,
            next_state=transition.next_state,
            done=transition.done,
        )
        self.total_updates += 1
        return {
            "real_td_error": 0.0,
            "planning_td_error": abs(td_error),
            "exploration_bonus": bonus,
            "model_buffer_size": float(len(self.world_model)),
        }

    def episode_ended(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "total_updates": self.total_updates,
                "world_model": self.world_model.state_dict(),
                "rng_state": self.rng.bit_generator.state,
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> DynaQAgent:
        payload = torch.load(Path(path), weights_only=False)
        agent = cls(q_table=payload["q_table"], **kwargs)
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        agent.total_updates = int(payload.get("total_updates", 0))
        agent.world_model.load_state_dict(payload.get("world_model", {}))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            agent.rng.bit_generator.state = rng_state
        return agent

    def _q_update(
        self,
        *,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> float:
        state_action = (*state, action)
        bootstrap = 0.0 if done else float(torch.max(self.q_table[next_state]).item())
        # Real or planned TD target: r + gamma * max_a' Q(s', a').
        target = float(reward) + self.gamma * bootstrap
        error = target - float(self.q_table[state_action].item())
        self.q_table[state_action] += self.alpha * error
        return float(error)

    def _planning_bonus(self, last_seen: int) -> float:
        del last_seen
        return 0.0


class DynaQPlusAgent(DynaQAgent):
    """Dyna-Q+ adds a planning-time exploration bonus based on time since visit."""

    def __init__(self, *args: Any, kappa: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kappa = kappa

    def save(self, path: str | Path) -> Path:
        output_path = super().save(path)
        payload = torch.load(output_path, weights_only=False)
        payload["kappa"] = self.kappa
        torch.save(payload, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> DynaQPlusAgent:
        payload = torch.load(Path(path), weights_only=False)
        agent = cls(q_table=payload["q_table"], kappa=payload.get("kappa", 1e-3), **kwargs)
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        agent.total_updates = int(payload.get("total_updates", 0))
        agent.world_model.load_state_dict(payload.get("world_model", {}))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            agent.rng.bit_generator.state = rng_state
        return agent

    def learn_real_transition(
        self,
        *,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> dict[str, float]:
        metrics = super().learn_real_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.world_model.ensure_all_actions_for_state(
            state,
            self.action_count,
            time_step=self.total_updates - 1,
        )
        metrics["model_buffer_size"] = float(len(self.world_model))
        return metrics

    def _planning_bonus(self, last_seen: int) -> float:
        # tau counts Q-updates (real + planning) since (s, a) was last seen, not
        # raw env steps — this mirrors the gold-standard notebook 09. Keep it
        # aligned with the notebook rather than switching to real-step-only.
        tau = max(0, self.total_updates - last_seen)
        # Dyna-Q+ bonus: kappa * sqrt(tau) favors actions left unexplored for longer.
        return float(self.kappa * np.sqrt(float(tau)))


class DeepDynaAgent(BaseAgent):
    """A compact Deep Dyna agent built from DQN plus a learned dynamics model."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        model_lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        model_train_steps: int = 1,
        imagined_updates: int = 4,
        start_learning_after: int = 1_000,
        rng: np.random.Generator | None = None,
        device: str = "auto",
        state_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        done_loss_weight: float = 1.0,
    ) -> None:
        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.model_train_steps = model_train_steps
        self.imagined_updates = imagined_updates
        self.start_learning_after = start_learning_after
        self.rng = rng or np.random.default_rng()
        self.update_steps = 0
        # Per-component loss weights for the dynamics model. The three losses live on
        # heterogeneous scales (MSE for continuous next-state and reward vs
        # BCE-with-logits for the done flag), so explicit weights make the trade-off
        # visible and tunable. Defaults of 1.0 preserve the original behaviour.
        self.state_loss_weight = state_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.done_loss_weight = done_loss_weight

        self.q_online = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()
        self.dynamics_model = NeuralDynamicsModel(
            obs_dim, n_actions, hidden_dim=hidden_dim
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)
        self.model_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=model_lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, rng=self.rng)

    def select_action(self, observation: Any, *, deterministic: bool = False) -> int:
        if not deterministic and float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        obs_tensor = self._obs_tensor(observation)
        with torch.no_grad():
            q_values = self.q_online(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        self.replay_buffer.push(obs, int(action), float(reward), next_obs, bool(done))

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        del kwargs
        if len(self.replay_buffer) < max(self.batch_size, self.start_learning_after):
            return self._empty_metrics()

        real_metrics = self.learn_q_from_real_batch(self.sample_replay_batch())
        model_metrics = self._train_model_for_real_batches()
        imagined_metrics = self._train_q_for_imagined_batches()
        return {
            "real_td_error": real_metrics["real_td_error"],
            "planning_td_error": imagined_metrics["planning_td_error"],
            "exploration_bonus": 0.0,
            "model_buffer_size": float(len(self.replay_buffer)),
            "q_loss": real_metrics["q_loss"],
            "model_prediction_loss": model_metrics["model_prediction_loss"],
            "reward_prediction_loss": model_metrics["reward_prediction_loss"],
            "done_prediction_loss": model_metrics["done_prediction_loss"],
            "imagined_update_count": imagined_metrics["imagined_update_count"],
        }

    def episode_ended(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_online": self.q_online.state_dict(),
                "q_target": self.q_target.state_dict(),
                "dynamics_model": self.dynamics_model.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "model_optimizer": self.model_optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_steps": self.update_steps,
                "replay_buffer": self.replay_buffer.state_dict(),
                "rng_state": self.rng.bit_generator.state,
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> DeepDynaAgent:
        payload = torch.load(Path(path), weights_only=False)
        agent = cls(**kwargs)
        agent.q_online.load_state_dict(payload["q_online"])
        agent.q_target.load_state_dict(payload["q_target"])
        agent.dynamics_model.load_state_dict(payload["dynamics_model"])
        agent.q_optimizer.load_state_dict(payload["q_optimizer"])
        agent.model_optimizer.load_state_dict(payload["model_optimizer"])
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        agent.update_steps = int(payload.get("update_steps", 0))
        agent.replay_buffer.load_state_dict(payload.get("replay_buffer", {}))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            agent.rng.bit_generator.state = rng_state
        return agent

    def sample_replay_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.replay_buffer.sample(self.batch_size, rng=self.rng)
        return tuple(t.to(self.device) for t in batch)

    def learn_q_from_real_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        q_loss, td_error = self._train_q_batch(batch)
        return {"q_loss": q_loss, "real_td_error": td_error}

    def learn_model_from_real_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        # Dynamics target: learn (s, a) -> (r, s', done).
        predicted_next, predicted_reward, predicted_done_logits = self.dynamics_model(
            states, actions
        )
        state_loss = F.mse_loss(predicted_next, next_states)
        reward_loss = F.mse_loss(predicted_reward, rewards)
        done_loss = F.binary_cross_entropy_with_logits(predicted_done_logits, dones)
        # Weighted sum: weights expose the heterogeneous-scale trade-off between MSE
        # losses (next-state, reward) and BCE-with-logits (done flag). Defaults=1.0.
        loss = (
            self.state_loss_weight * state_loss
            + self.reward_loss_weight * reward_loss
            + self.done_loss_weight * done_loss
        )
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return {
            "model_prediction_loss": float(state_loss.item()),
            "reward_prediction_loss": float(reward_loss.item()),
            "done_prediction_loss": float(done_loss.item()),
        }

    def learn_q_from_imagined_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        states, _, _, _, _ = batch
        with torch.no_grad():
            imagined_actions = self.q_online(states).argmax(dim=1)
            imagined_next, imagined_reward, imagined_done_logits = self.dynamics_model(
                states, imagined_actions
            )
            imagined_done = (torch.sigmoid(imagined_done_logits) > 0.5).float()
        # Imagined update bootstraps Q from model-generated transitions.
        _, td_error = self._train_q_batch(
            (states, imagined_actions, imagined_reward, imagined_next, imagined_done)
        )
        return td_error

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "real_td_error": 0.0,
            "planning_td_error": 0.0,
            "exploration_bonus": 0.0,
            "model_buffer_size": float(len(self.replay_buffer)),
            "q_loss": 0.0,
            "model_prediction_loss": 0.0,
            "reward_prediction_loss": 0.0,
            "done_prediction_loss": 0.0,
            "imagined_update_count": 0.0,
        }

    def _train_model_for_real_batches(self) -> dict[str, float]:
        totals = {
            "model_prediction_loss": 0.0,
            "reward_prediction_loss": 0.0,
            "done_prediction_loss": 0.0,
        }
        for _ in range(self.model_train_steps):
            metrics = self.learn_model_from_real_batch(self.sample_replay_batch())
            for key, value in metrics.items():
                totals[key] += value
        if self.model_train_steps == 0:
            return totals
        return {key: value / self.model_train_steps for key, value in totals.items()}

    def _train_q_for_imagined_batches(self) -> dict[str, float]:
        td_errors: list[float] = []
        for _ in range(self.imagined_updates):
            td_errors.append(self.learn_q_from_imagined_batch(self.sample_replay_batch()))
        return {
            "planning_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
            "imagined_update_count": float(len(td_errors)),
        }

    def _train_q_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[float, float]:
        states, actions, rewards, next_states, dones = batch
        q_values = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(next_states).max(dim=1).values
            # Bellman target for Deep Dyna Q updates.
            targets = rewards + self.gamma * next_q * (1.0 - dones)
        td_errors = targets - q_values
        # Bellman loss on real or imagined transitions.
        loss = F.mse_loss(q_values, targets)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())
        return float(loss.item()), float(td_errors.abs().mean().item())

    def _obs_tensor(self, obs: Any) -> torch.Tensor:
        tensor = torch.as_tensor(obs, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
