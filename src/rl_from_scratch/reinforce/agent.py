"""REINFORCE and REINFORCE with baseline agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.reinforce.network import PolicyNetwork, ValueNetwork


class ReinforceAgent(BaseAgent):
    """REINFORCE agent (Monte Carlo Policy Gradient).

    Collects a full trajectory per episode, then performs a policy
    gradient update using the Monte Carlo returns.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    n_actions:
        Number of discrete actions.
    config:
        Optional config dataclass; the hyperparameters are extracted
        from it if provided.
    hidden_dim:
        Width of each hidden layer of the policy network.
    lr:
        Learning rate of the Adam optimizer.
    gamma:
        Discount factor.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "auto",
    ) -> None:
        from rl_from_scratch.core.utils import resolve_device
        self.device = torch.device(resolve_device(device))

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma

        self.policy_net = PolicyNetwork(obs_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Episode buffers
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

    # ------------------------------------------------------------------
    # Interface BaseAgent
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> int:
        """Choose an action according to the current policy.

        In stochastic mode (default), we sample from the categorical
        distribution. In deterministic mode, we take the argmax.
        The action's log-prob is stored for the update.

        Parameters
        ----------
        observation:
            Current observation from the environment.
        deterministic:
            If True, we take the most likely action (argmax).
        """
        obs_t = self._to_tensor(observation)
        logits = self.policy_net(obs_t)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=1)
        else:
            action = dist.sample()
            self.log_probs.append(dist.log_prob(action))

        return int(action.item())

    def store_transition(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Store the reward of the current step."""
        self.rewards.append(float(reward))

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Compute the Monte Carlo returns and update the policy.

        Called once at the end of an episode. Clears the buffers after the
        update.

        Returns
        -------
        dict
            ``{"policy_loss": float}``
        """
        if not self.rewards:
            return {}

        returns = self._compute_returns()
        loss = self._compute_policy_loss(returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._clear_episode()
        return {"policy_loss": loss.item()}

    def episode_ended(self) -> None:
        """No action required at the end of an episode (no epsilon decay)."""

    def save(self, path: str | Path) -> Path:
        """Save the state dict of the policy network."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy_net": self.policy_net.state_dict()}, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        obs_dim: int,
        n_actions: int,
        **kwargs: Any,
    ) -> ReinforceAgent:
        """Load an agent from a saved checkpoint."""
        agent = cls(obs_dim, n_actions, **kwargs)
        checkpoint = torch.load(Path(path), weights_only=True)
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        return agent

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_returns(self) -> torch.Tensor:
        """Compute the normalized Monte Carlo returns.

        Iterates over the rewards backwards to accumulate G = r + γG, then
        normalizes (zero mean, unit standard deviation) to stabilize
        training.

        Returns
        -------
        torch.Tensor
            Normalized returns of shape ``(T,)``.
        """
        returns: list[float] = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        return returns_t

    def _compute_policy_loss(self, returns: torch.Tensor) -> torch.Tensor:
        """Compute the policy loss: -(log π(a|s) * G).

        Parameters
        ----------
        returns:
            Returns (possibly normalized) of shape ``(T,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss (mean over the episode).
        """
        log_probs_t = torch.cat(self.log_probs)
        return -(log_probs_t * returns).mean()

    def _clear_episode(self) -> None:
        """Clear the log-probs and rewards buffers."""
        self.log_probs = []
        self.rewards = []


class ReinforceBaselineAgent(ReinforceAgent):
    """REINFORCE with a learned value baseline (Monte Carlo Actor-Critic).

    Extends ``ReinforceAgent`` by adding a value network V(s) trained
    in parallel. The advantages A = G - V(s) replace the raw returns
    in the policy loss, which reduces variance.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    n_actions:
        Number of discrete actions.
    config:
        Optional config dataclass; ``lr_policy`` and ``lr_value`` are
        extracted from it if provided.
    hidden_dim:
        Width of each hidden layer.
    lr_policy:
        Learning rate for the policy.
    lr_value:
        Learning rate for the value network.
    gamma:
        Discount factor.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dim: int = 64,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        device: str = "auto",
    ) -> None:
        # Initialize the policy network via the parent
        # We map lr_policy → lr for ReinforceAgent
        super().__init__(
            obs_dim,
            n_actions,
            hidden_dim=hidden_dim,
            lr=lr_policy,
            gamma=gamma,
            device=device,
        )

        # Value network and its optimizer
        self.value_net = ValueNetwork(obs_dim, hidden_dim=hidden_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr_value
        )

        # Buffer of the episode's states
        self.states: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Any, *, deterministic: bool = False
    ) -> int:
        """Memorize the current state, then delegate to the parent policy."""
        if not deterministic:
            obs_t = self._to_tensor(observation)
            self.states.append(obs_t)
        return super().select_action(observation, deterministic=deterministic)

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        """Update the policy and the value network.

        1. Compute the raw returns G (unnormalized) — targets for V(s).
        2. Compute the values V(s) for each state of the episode.
        3. Advantages A = G - V(s).detach(), normalized.
        4. Policy loss: -(log π(a|s) * A).mean()
        5. Value loss: MSE(V(s), G).
        6. Two separate optimization steps.

        Returns
        -------
        dict
            ``{"policy_loss": float, "value_loss": float}``
        """
        if not self.rewards:
            return {}

        # Raw (unnormalized) returns for the value target
        raw_returns = self._compute_raw_returns()

        # Values V(s) for the episode's states
        states_t = torch.cat(self.states, dim=0)  # (T, obs_dim)
        values = self.value_net(states_t)          # (T,)

        # Normalized advantages
        advantages = raw_returns - values.detach()
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss = self._compute_policy_loss(advantages)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Value loss
        value_loss = F.mse_loss(values, raw_returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self._clear_episode()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def _clear_episode(self) -> None:
        """Clear the log-probs, rewards, and states buffers."""
        super()._clear_episode()
        self.states = []

    def save(self, path: str | Path) -> Path:
        """Save the state dicts of the policy and the value network."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "value_net": self.value_net.state_dict(),
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
    ) -> ReinforceBaselineAgent:
        """Load an agent from a saved checkpoint."""
        agent = cls(obs_dim, n_actions, **kwargs)
        checkpoint = torch.load(Path(path), weights_only=True)
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.value_net.load_state_dict(checkpoint["value_net"])
        return agent

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_raw_returns(self) -> torch.Tensor:
        """Compute the *unnormalized* Monte Carlo returns.

        Used as regression targets for the value network.

        Returns
        -------
        torch.Tensor
            Raw returns of shape ``(T,)``.
        """
        returns: list[float] = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
