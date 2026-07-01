"""Neural network architectures for DQN-family agents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple fully-connected Q-network.

    Maps observations to Q-values for each discrete action.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    n_actions:
        Number of discrete actions.
    hidden_dim:
        Width of each hidden layer.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for the given observations.

        Parameters
        ----------
        x:
            Observation tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Q-values of shape ``(batch, n_actions)``.
        """
        return self.net(x)


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer used by Rainbow DQN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.resample_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features**0.5)
        self.bias_sigma.data.fill_(self.std_init / self.out_features**0.5)

    def _scale_noise(self, size: int) -> torch.Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign() * noise.abs().sqrt()

    def resample_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingQNetwork(nn.Module):
    """Dueling network with optional noisy linear heads."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        *,
        noisy: bool = False,
        noisy_std_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        linear_cls: type[nn.Linear] | type[NoisyLinear]
        linear_cls = NoisyLinear if noisy else nn.Linear
        linear_kwargs = {"std_init": noisy_std_init} if noisy else {}
        self.value = nn.Sequential(
            linear_cls(hidden_dim, hidden_dim, **linear_kwargs),
            nn.ReLU(),
            linear_cls(hidden_dim, 1, **linear_kwargs),
        )
        self.advantage = nn.Sequential(
            linear_cls(hidden_dim, hidden_dim, **linear_kwargs),
            nn.ReLU(),
            linear_cls(hidden_dim, n_actions, **linear_kwargs),
        )

    def resample_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.resample_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class CategoricalDuelingQNetwork(nn.Module):
    """Dueling C51 network that outputs categorical action-value distributions."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        *,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
        noisy_std_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        linear_cls: type[nn.Linear] | type[NoisyLinear]
        linear_cls = NoisyLinear if noisy else nn.Linear
        linear_kwargs = {"std_init": noisy_std_init} if noisy else {}
        self.value = nn.Sequential(
            linear_cls(hidden_dim, hidden_dim, **linear_kwargs),
            nn.ReLU(),
            linear_cls(hidden_dim, n_atoms, **linear_kwargs),
        )
        self.advantage = nn.Sequential(
            linear_cls(hidden_dim, hidden_dim, **linear_kwargs),
            nn.ReLU(),
            linear_cls(hidden_dim, n_actions * n_atoms, **linear_kwargs),
        )
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

    def resample_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.resample_noise()

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value(features).view(-1, 1, self.n_atoms)
        advantage = self.advantage(features).view(-1, self.n_actions, self.n_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return probs.clamp(min=1e-6)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.dist(x)
        return torch.sum(probs * self.support.view(1, 1, -1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dist(x)
