"""DQN-family agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.deep_q.buffer import NStepTransitionAccumulator, PrioritizedReplayBuffer, ReplayBuffer
from rl_from_scratch.deep_q.network import CategoricalDuelingQNetwork, QNetwork


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and a target network.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    n_actions:
        Number of discrete actions.
    config:
        Optional config dataclass; when provided, hyperparameters are
        read from it.
    hidden_dim:
        Width of each hidden layer in the Q-network.
    lr:
        Learning rate for the Adam optimiser.
    gamma:
        Discount factor.
    epsilon:
        Initial exploration probability for epsilon-greedy.
    epsilon_decay:
        Multiplicative decay applied to epsilon at each episode end.
    min_epsilon:
        Floor for epsilon decay.
    buffer_capacity:
        Maximum size of the replay buffer.
    batch_size:
        Number of transitions sampled per learning step.
    target_update_freq:
        Copy online weights to target every this many learning steps.
    rng:
        NumPy random generator for epsilon-greedy tie-breaking.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        rng: np.random.Generator | None = None,
        device: str = "auto",
    ) -> None:
        from rl_from_scratch.core.utils import resolve_device
        self.device = torch.device(resolve_device(device))

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = rng or np.random.default_rng()
        self.steps = 0

        self.q_online = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> int:
        """Choose an action via epsilon-greedy over the online Q-network."""
        if not deterministic and float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        obs_t = self._to_tensor(observation)
        with torch.no_grad():
            q_values = self.q_online(obs_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self.buffer.push(obs, int(action), reward, next_obs, done)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Sample a batch, compute loss, and update the online network.

        Returns an empty dict when the buffer has fewer samples than
        ``batch_size``.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        batch = tuple(t.to(self.device) for t in self.buffer.sample(self.batch_size))
        loss = self._compute_loss(batch)  # type: ignore[arg-type]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        return {"loss": loss.item()}

    def episode_ended(self) -> None:
        """Decay epsilon at the end of an episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        """Save the online Q-network state dict."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_online.state_dict(), output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        obs_dim: int,
        n_actions: int,
        **kwargs: Any,
    ) -> DQNAgent:
        """Load an agent from a saved checkpoint."""
        agent = cls(obs_dim, n_actions, **kwargs)
        state_dict = torch.load(Path(path), weights_only=True)
        agent.q_online.load_state_dict(state_dict)
        agent.q_target.load_state_dict(state_dict)
        return agent

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the standard DQN loss (Bellman MSE).

        Target: ``r + gamma * max_a' Q_target(s', a') * (1 - done)``.
        """
        states, actions, rewards, next_states, dones = batch

        q_values = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_max = self.q_target(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_max * (1.0 - dones)

        return torch.nn.functional.mse_loss(q_values, targets)


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent — decouples action selection from evaluation.

    The only difference from vanilla DQN is the loss computation: the
    *online* network selects the best next action, but the *target*
    network evaluates its Q-value.  This reduces over-estimation bias.
    """

    def _compute_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Double DQN loss.

        Target: ``r + gamma * Q_target(s', argmax_a' Q_online(s', a')) * (1 - done)``.
        """
        states, actions, rewards, next_states, dones = batch

        q_values = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.q_online(next_states).argmax(dim=1)
            next_q = self.q_target(next_states).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        return torch.nn.functional.mse_loss(q_values, targets)


class RainbowDQNAgent(BaseAgent):
    """Rainbow DQN agent with noisy nets, PER, n-step returns, and C51."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        n_steps: int = 3,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy_std_init: float = 0.5,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_beta_steps: int = 100_000,
        priority_eps: float = 1e-6,
        rng: np.random.Generator | None = None,
        device: str = "auto",
    ) -> None:
        from rl_from_scratch.core.utils import resolve_device

        self.device = torch.device(resolve_device(device))
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_steps = n_steps
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.priority_eps = priority_eps
        self.epsilon = 0.0
        self.steps = 0

        self.q_online = CategoricalDuelingQNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            noisy=True,
            noisy_std_init=noisy_std_init,
        ).to(self.device)
        self.q_target = CategoricalDuelingQNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            noisy=True,
            noisy_std_init=noisy_std_init,
        ).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(
            buffer_capacity,
            alpha=priority_alpha,
            beta=priority_beta,
            beta_annealing_steps=priority_beta_steps,
            eps=priority_eps,
        )
        self.n_step_accumulator = NStepTransitionAccumulator(n_steps=n_steps, gamma=gamma)
        self.support = self.q_online.support.to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.gamma_n = gamma**n_steps

    def select_action(self, observation: Any, *, deterministic: bool = False) -> int:
        obs_t = self._to_tensor(observation)
        was_training = self.q_online.training
        with torch.no_grad():
            if deterministic:
                self.q_online.eval()
            else:
                self.q_online.train()
                self.q_online.resample_noise()
            q_values = self.q_online.q_values(obs_t)
        self.q_online.train(was_training)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        aggregated = self.n_step_accumulator.push(obs, int(action), reward, next_obs, done)
        for transition in aggregated:
            self.buffer.push(*transition)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        self.q_online.train()
        self.q_online.resample_noise()
        dist = self.q_online.dist(states)
        chosen_dist = dist[torch.arange(states.size(0), device=self.device), actions]
        chosen_dist = chosen_dist.clamp(min=1e-6)
        q_values = torch.sum(chosen_dist * self.support.view(1, -1), dim=1)

        with torch.no_grad():
            self.q_target.train()
            self.q_target.resample_noise()
            next_actions = self.q_online.q_values(next_states).argmax(dim=1)
            next_dist_all = self.q_target.dist(next_states)
            next_dist = next_dist_all[
                torch.arange(next_states.size(0), device=self.device),
                next_actions,
            ]
            target_dist = self._project_distribution(next_dist, rewards, dones)
            next_q = torch.sum(next_dist * self.support.view(1, -1), dim=1)
            target_q = rewards + self.gamma_n * next_q * (1.0 - dones)
            td_error = (target_q - q_values).abs()

        per_sample_loss = -(target_dist * torch.log(chosen_dist)).sum(dim=1)
        loss = torch.mean(weights * per_sample_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.update_priorities(indices, (td_error.detach().cpu().numpy() + self.priority_eps))

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_values.mean().item()),
            "td_error_mean": float(td_error.mean().item()),
            "beta": float(self.buffer.beta),
        }

    def _project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rewards.size(0)
        projected = torch.zeros(batch_size, self.n_atoms, device=next_dist.device)

        tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.gamma_n * self.support.view(1, -1)
        tz = tz.clamp(self.v_min, self.v_max)
        b = (tz - self.v_min) / self.delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        for atom in range(self.n_atoms):
            lower_idx = lower[:, atom]
            upper_idx = upper[:, atom]
            mass = next_dist[:, atom]

            same = lower_idx == upper_idx
            projected[torch.arange(batch_size, device=next_dist.device), lower_idx] += mass * same.float()

            lower_weight = (upper_idx.float() - b[:, atom]) * (~same).float()
            upper_weight = (b[:, atom] - lower_idx.float()) * (~same).float()

            projected[torch.arange(batch_size, device=next_dist.device), lower_idx] += mass * lower_weight
            projected[torch.arange(batch_size, device=next_dist.device), upper_idx] += mass * upper_weight

        projected = projected / projected.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return projected

    def episode_ended(self) -> None:
        for transition in self.n_step_accumulator.flush():
            self.buffer.push(*transition)
        self.q_online.resample_noise()

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_online": self.q_online.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "buffer": self.buffer.state_dict(),
                "n_step_accumulator": self.n_step_accumulator.state_dict(),
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        obs_dim: int,
        n_actions: int,
        **kwargs: Any,
    ) -> RainbowDQNAgent:
        agent = cls(obs_dim, n_actions, **kwargs)
        state = torch.load(Path(path), weights_only=False, map_location=agent.device)
        if "q_online" in state:
            agent.q_online.load_state_dict(state["q_online"])
            agent.q_target.load_state_dict(state.get("q_target", state["q_online"]))
            if "optimizer" in state:
                agent.optimizer.load_state_dict(state["optimizer"])
            agent.steps = int(state.get("steps", agent.steps))
            if "buffer" in state:
                agent.buffer.load_state_dict(state["buffer"])
            if "n_step_accumulator" in state:
                agent.n_step_accumulator.load_state_dict(state["n_step_accumulator"])
        else:
            agent.q_online.load_state_dict(state)
            agent.q_target.load_state_dict(state)
        return agent
