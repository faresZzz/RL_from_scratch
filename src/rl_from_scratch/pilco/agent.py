"""PILCO and Deep PILCO agents."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.env import encode_obs
from rl_from_scratch.pilco.bnn import (
    BayesianDynamicsNetwork,
    predict_trajectory_particles,
    train_bnn_on_buffer,
)
from rl_from_scratch.pilco.buffer import TransitionBuffer
from rl_from_scratch.pilco.cost import (
    expected_inverted_pendulum_cost,
    inverted_pendulum_particle_cost,
)
from rl_from_scratch.pilco.gp import MultiOutputGP
from rl_from_scratch.pilco.policy import build_policy
from rl_from_scratch.pilco.belief_propagation import predict_trajectory


def _state_target(obs_dim: int, env_obs_dim: int, encode_angle_flag: bool, dtype: torch.dtype) -> Tensor:
    if encode_angle_flag and env_obs_dim == 4:
        return torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=dtype)
    if obs_dim == 3:
        return torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    return torch.zeros(obs_dim, dtype=dtype)


def _state_weight(cost_weight: tuple, obs_dim: int, dtype: torch.dtype) -> Tensor:
    w_diag = list(cost_weight)
    if len(w_diag) < obs_dim:
        w_diag = w_diag + [1.0] * (obs_dim - len(w_diag))
    elif len(w_diag) > obs_dim:
        w_diag = w_diag[:obs_dim]
    return torch.diag(torch.tensor(w_diag, dtype=dtype))


class PilcoAgent(BaseAgent):
    """GP dynamics + analytic belief propagation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        action_low: Any,
        action_high: Any,
        n_basis: int = 50,
        horizon: int = 30,
        gp_fit_steps: int = 50,
        policy_opt_steps: int = 50,
        policy_lr: float = 0.3,
        n_initial_beliefs: int = 1,
        max_gp_points: int = 300,
        init_state_cov: float = 0.01,
        cost_weight: tuple = (1.0, 1.0, 0.1),
        exploration_noise: float = 0.0,
        encode_angle: bool = False,
        fixed_horizon_steps: int = 0,
        policy_type: str = "rbf",
        policy_hidden_dim: int = 64,
        policy_hidden_layers: int = 2,
        cost_mode: str = "saturating",
        terminal_penalty: float = 10.0,
        action_cost_weight: float = 1e-4,
        validation_fraction: float = 0.15,
        validation_min_points: int = 32,
        gamma: float = 0.99,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        del gamma, device, action_low
        self.env_obs_dim = int(obs_dim)
        self.encode_angle = bool(encode_angle)
        self.fixed_horizon_steps = int(fixed_horizon_steps)
        self.obs_dim = self.env_obs_dim + 1 if self.encode_angle else self.env_obs_dim
        self.action_dim = int(action_dim)
        self.n_basis = int(n_basis)
        self.horizon = int(horizon)
        self.gp_fit_steps = int(gp_fit_steps)
        self.policy_opt_steps = int(policy_opt_steps)
        self.policy_lr = float(policy_lr)
        self.n_initial_beliefs = int(n_initial_beliefs)
        self.max_gp_points = int(max_gp_points)
        self.init_state_cov = float(init_state_cov)
        self.cost_weight_tuple = tuple(cost_weight)
        self.exploration_noise = float(exploration_noise)
        self.policy_type = str(policy_type)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.policy_hidden_layers = int(policy_hidden_layers)
        self.cost_mode = str(cost_mode)
        self.terminal_penalty = float(terminal_penalty)
        self.action_cost_weight = float(action_cost_weight)
        self.validation_fraction = float(validation_fraction)
        self.validation_min_points = int(validation_min_points)
        self.rng = np.random.default_rng(seed)

        action_high_t = torch.tensor(np.asarray(action_high, dtype=np.float64).flatten(), dtype=torch.float64)
        self.policy = build_policy(
            self.policy_type,
            self.obs_dim,
            action_dim,
            action_high=action_high_t,
            n_basis=n_basis,
            hidden_dim=self.policy_hidden_dim,
            hidden_layers=self.policy_hidden_layers,
        ).double()
        self.gp: MultiOutputGP | None = None
        self._dynamics_fit_count = 0
        self.buffer = TransitionBuffer(max_gp_points=max_gp_points, rng=self.rng)
        self.target = _state_target(self.obs_dim, self.env_obs_dim, self.encode_angle, torch.float64)
        self.weight = _state_weight(cost_weight, self.obs_dim, torch.float64)
        self._optimization_beliefs: list[tuple[Tensor, Tensor]] | None = None

    def _encode(self, obs: Any) -> np.ndarray:
        raw = np.asarray(obs, dtype=np.float64)
        return encode_obs(raw) if self.encode_angle else raw

    def _is_failure(self, next_obs: Any, done: bool) -> bool:
        if done:
            return True
        if self.encode_angle and self.env_obs_dim == 4:
            raw = np.asarray(next_obs, dtype=np.float64)
            return bool(not np.isfinite(raw).all() or abs(raw[1]) > 0.2)
        return False

    def _step_cost_fn(self):
        if self.cost_mode != "inverted_pendulum":
            return None
        return lambda mu_x, sigma_x, mu_u, sigma_u: expected_inverted_pendulum_cost(
            mu_x,
            sigma_x,
            mu_u,
            sigma_u,
            state_weight=self.weight,
            target=self.target,
            terminal_penalty=self.terminal_penalty,
            action_cost_weight=self.action_cost_weight,
        )

    def select_action(self, obs: Any, *, deterministic: bool = False) -> np.ndarray:
        encoded = self._encode(obs)
        obs_t = torch.tensor(encoded, dtype=torch.float64)
        with torch.no_grad():
            action_t = self.policy.forward(obs_t)
        action = np.nan_to_num(action_t.numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if not deterministic and self.exploration_noise > 0.0:
            action = action + self.rng.normal(0.0, self.exploration_noise, size=action.shape).astype(np.float32)
        return action

    def store_transition(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        del reward
        self.buffer.push(
            self._encode(obs),
            action,
            self._encode(next_obs),
            failure=self._is_failure(next_obs, done),
        )

    def fit_dynamics(self) -> tuple[float, float]:
        train_x, train_y, holdout_x, holdout_y, _ = self.buffer.get_train_and_holdout_tensors(
            validation_fraction=self.validation_fraction,
            validation_min_points=self.validation_min_points,
            seed=0,
        )
        finite_rows = torch.isfinite(train_x).all(dim=1) & torch.isfinite(train_y).all(dim=1)
        train_x, train_y = train_x[finite_rows], train_y[finite_rows]
        if self.gp is None:
            self.gp = MultiOutputGP(train_x.shape[1], train_y.shape[1]).double()
            self.gp.initialize_from_data(train_x, train_y)
        else:
            self.gp.set_data(train_x, train_y)
        fit_steps = self.gp_fit_steps if self._dynamics_fit_count == 0 else max(6, self.gp_fit_steps // 3)
        nlml_list = self.gp.fit(train_x, train_y, n_steps=fit_steps)
        self._dynamics_fit_count += 1
        holdout_mse = 0.0
        if len(holdout_x) > 0:
            with torch.no_grad():
                pred, _ = self.gp.predict(holdout_x)
                holdout_mse = float(((pred - holdout_y) ** 2).mean().item())
        return float(np.mean([v for v in nlml_list if math.isfinite(v)])), holdout_mse

    def set_optimization_beliefs(self, beliefs: list[tuple[Tensor, Tensor]]) -> None:
        self._optimization_beliefs = [
            (mu.detach().clone().to(dtype=torch.float64), sigma.detach().clone().to(dtype=torch.float64))
            for mu, sigma in beliefs
        ]

    def optimize_policy(self) -> tuple[float, float]:
        if self.gp is None:
            raise RuntimeError("Call fit_dynamics() before optimize_policy().")
        if self._optimization_beliefs:
            beliefs = [
                (mu, sigma)
                for mu, sigma in self._optimization_beliefs[: self.n_initial_beliefs]
            ]
        else:
            initial_means = torch.tensor(
                self.buffer.initial_state_samples(self.n_initial_beliefs),
                dtype=torch.float64,
            )
            cov_diag = torch.tensor(self.buffer.initial_state_cov(), dtype=torch.float64).clamp_min(self.init_state_cov)
            sigma0 = torch.diag(cov_diag)
            beliefs = [(mu0, sigma0) for mu0 in initial_means]
        for p in self.gp.parameters():
            p.requires_grad_(False)
        for p in self.policy.parameters():
            p.requires_grad_(True)
        # K^-1 is constant while the GP is frozen; computing it inside every
        # horizon step would dominate L-BFGS without changing the objective.
        k_invs = [torch.cholesky_inverse(g._L).detach() for g in self.gp.gps]

        def trajectory_cost() -> float:
            with torch.no_grad():
                costs = [
                    predict_trajectory(
                        self.gp,
                        self.policy,
                        mu0,
                        sigma,
                        horizon=self.horizon,
                        target=self.target,
                        weight=self.weight,
                        project_encoded_angle=self.encode_angle,
                        step_cost_fn=self._step_cost_fn(),
                        k_invs=k_invs,
                    )[0]
                    for mu0, sigma in beliefs
                ]
            return float(torch.stack(costs).mean().item())

        optimizer = torch.optim.LBFGS(
            self.policy.parameters(),
            lr=self.policy_lr,
            max_iter=self.policy_opt_steps,
            line_search_fn="strong_wolfe",
        )
        backup = [p.detach().clone() for p in self.policy.parameters()]
        before = trajectory_cost()

        def restore() -> None:
            with torch.no_grad():
                for param, saved in zip(self.policy.parameters(), backup):
                    param.copy_(saved)

        def closure() -> Tensor:
            optimizer.zero_grad()
            costs = [
                predict_trajectory(
                    self.gp,
                    self.policy,
                    mu0,
                    sigma,
                    horizon=self.horizon,
                    target=self.target,
                    weight=self.weight,
                    project_encoded_angle=self.encode_angle,
                    step_cost_fn=self._step_cost_fn(),
                    k_invs=k_invs,
                )[0]
                for mu0, sigma in beliefs
            ]
            cost = torch.stack(costs).mean()
            cost.backward()
            return cost

        try:
            optimizer.step(closure)
        except (torch.linalg.LinAlgError, RuntimeError):
            restore()

        if not all(bool(torch.isfinite(p).all()) for p in self.policy.parameters()):
            restore()
        after = trajectory_cost()
        if (not math.isfinite(after)) or after > before:
            restore()
            after = trajectory_cost()
        for p in self.gp.parameters():
            p.requires_grad_(True)
        return before, after if math.isfinite(after) else before

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        del kwargs
        nll, holdout_mse = self.fit_dynamics()
        predicted_before, predicted_after = self.optimize_policy()
        return {
            "model_nll": float(nll) if math.isfinite(nll) else 0.0,
            "holdout_mse": float(holdout_mse) if math.isfinite(holdout_mse) else 0.0,
            "predicted_cost_before": float(predicted_before),
            "predicted_cost": float(predicted_after) if math.isfinite(predicted_after) else float(predicted_before),
            "gp_points": float(len(self.buffer)),
        }

    def episode_ended(self) -> None:
        return None

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "buffer": self.buffer.state_dict(),
                "gp": self.gp.state_dict() if self.gp is not None else None,
                "target": self.target,
                "weight": self.weight,
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "PilcoAgent":
        payload = torch.load(Path(path), weights_only=False)
        agent = cls(**kwargs)
        agent.policy.load_state_dict(payload["policy"])
        agent.buffer.load_state_dict(payload.get("buffer", {}))
        if payload.get("gp") is not None:
            agent.gp = MultiOutputGP(agent.obs_dim + agent.action_dim, agent.obs_dim).double()
            gp_state = payload["gp"]
            variable_buffers = {"X", "y", "_beta", "_L"}
            fixed_state = {
                k: v
                for k, v in gp_state.items()
                if not any(k.endswith("." + name) for name in variable_buffers)
            }
            agent.gp.load_state_dict(fixed_state, strict=False)
            for i, single_gp in enumerate(agent.gp.gps):
                prefix = f"gps.{i}."
                for name in variable_buffers:
                    key = prefix + name
                    if key in gp_state:
                        setattr(single_gp, name, gp_state[key])
        if "target" in payload:
            agent.target = payload["target"]
        if "weight" in payload:
            agent.weight = payload["weight"]
        return agent


class DeepPilcoAgent(BaseAgent):
    """BNN dynamics + particle rollouts."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        action_low: Any,
        action_high: Any,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout_p: float = 0.05,
        n_particles: int = 30,
        horizon: int = 25,
        model_train_steps: int = 100,
        model_batch_size: int = 64,
        model_lr: float = 1e-3,
        policy_opt_steps: int = 50,
        policy_lr: float = 0.01,
        n_basis: int = 50,
        max_gp_points: int = 300,
        init_state_cov: float = 0.01,
        cost_weight: tuple = (1.0, 1.0, 0.1),
        exploration_noise: float = 0.0,
        encode_angle: bool = False,
        fixed_horizon_steps: int = 0,
        policy_type: str = "rbf",
        policy_hidden_dim: int = 64,
        policy_hidden_layers: int = 2,
        cost_mode: str = "saturating",
        validation_fraction: float = 0.15,
        validation_min_points: int = 32,
        gamma: float = 0.99,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        del gamma, action_low
        self.env_obs_dim = int(obs_dim)
        self.encode_angle = bool(encode_angle)
        self.fixed_horizon_steps = int(fixed_horizon_steps)
        self.obs_dim = self.env_obs_dim + 1 if self.encode_angle else self.env_obs_dim
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout_p = float(dropout_p)
        self.n_particles = int(n_particles)
        self.horizon = int(horizon)
        self.model_train_steps = int(model_train_steps)
        self.model_batch_size = int(model_batch_size)
        self.model_lr = float(model_lr)
        self.policy_opt_steps = int(policy_opt_steps)
        self.policy_lr = float(policy_lr)
        self.n_basis = int(n_basis)
        self.max_gp_points = int(max_gp_points)
        self.init_state_cov = float(init_state_cov)
        self.cost_weight_tuple = tuple(cost_weight)
        self.exploration_noise = float(exploration_noise)
        self.policy_type = str(policy_type)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.policy_hidden_layers = int(policy_hidden_layers)
        self.cost_mode = str(cost_mode)
        self.validation_fraction = float(validation_fraction)
        self.validation_min_points = int(validation_min_points)
        self.rng = np.random.default_rng(seed)
        self.dtype = torch.float32
        if device == "auto":
            device = "cpu"

        input_dim = self.obs_dim + action_dim
        self.net = BayesianDynamicsNetwork(
            input_dim,
            self.obs_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_p=dropout_p,
        ).to(device=device, dtype=self.dtype)
        self._model_opt_state: dict[str, Any] | None = None

        self.action_high_t = torch.tensor(np.asarray(action_high, dtype=np.float32).flatten(), dtype=self.dtype)
        self.policy = build_policy(
            self.policy_type,
            self.obs_dim,
            action_dim,
            action_high=self.action_high_t,
            n_basis=n_basis,
            hidden_dim=self.policy_hidden_dim,
            hidden_layers=self.policy_hidden_layers,
        ).to(dtype=self.dtype)
        self.buffer = TransitionBuffer(max_gp_points=max_gp_points, rng=self.rng)
        self.target = _state_target(self.obs_dim, self.env_obs_dim, self.encode_angle, self.dtype)
        self.weight = _state_weight(cost_weight, self.obs_dim, self.dtype)
        self._reset_particles: Tensor | None = None

    def _encode(self, obs: Any) -> np.ndarray:
        raw = np.asarray(obs, dtype=np.float64)
        return encode_obs(raw).astype(np.float32) if self.encode_angle else raw.astype(np.float32)

    def _is_failure(self, next_obs: Any, done: bool) -> bool:
        if done:
            return True
        if self.encode_angle and self.env_obs_dim == 4:
            raw = np.asarray(next_obs, dtype=np.float64)
            return bool(not np.isfinite(raw).all() or abs(raw[1]) > 0.2)
        return False

    def _particle_step_cost_fn(self):
        if self.cost_mode != "inverted_pendulum":
            return None
        return lambda states, actions: inverted_pendulum_particle_cost(
            states,
            actions,
            state_weight=self.weight,
            target=self.target,
        )

    def select_action(self, obs: Any, *, deterministic: bool = False) -> np.ndarray:
        encoded = self._encode(obs)
        obs_t = torch.tensor(encoded, dtype=self.dtype)
        with torch.no_grad():
            action_t = self.policy.forward(obs_t)
        action = action_t.numpy().astype(np.float32)
        if not deterministic and self.exploration_noise > 0.0:
            action = action + self.rng.normal(0.0, self.exploration_noise, size=action.shape).astype(np.float32)
        return action

    def store_transition(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        del reward
        self.buffer.push(
            self._encode(obs),
            action,
            self._encode(next_obs),
            failure=self._is_failure(next_obs, done),
        )

    def _train_bnn(self, *, iteration: int = 0) -> tuple[float, dict[str, Any]]:
        all_x, all_y = self.buffer.get_recent_tensors(cap=self.max_gp_points)
        loss, metrics = train_bnn_on_buffer(
            self.net,
            all_x,
            all_y,
            n_steps=self.model_train_steps,
            batch_size=self.model_batch_size,
            lr=self.model_lr,
            seed=1000 + int(iteration),
        )
        return loss, metrics

    def set_reset_particles(self, particles: Tensor) -> None:
        self._reset_particles = particles.detach().clone().to(dtype=self.dtype)

    def _optimize_policy(self, *, iteration: int = 0) -> tuple[float, float]:
        for p in self.net.parameters():
            p.requires_grad_(False)
        for p in self.policy.parameters():
            p.requires_grad_(True)
        self.net.eval()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        backup = [p.detach().clone() for p in self.policy.parameters()]
        seed = 2000 + int(iteration)
        if self._reset_particles is not None:
            base_particles = self._reset_particles
        else:
            mu0 = torch.tensor(self.buffer.initial_state_mean(), dtype=self.dtype)
            cov_diag = torch.tensor(self.buffer.initial_state_cov(), dtype=self.dtype).clamp_min(self.init_state_cov)
            sigma0 = torch.diag(cov_diag)
            chol = torch.linalg.cholesky(sigma0 + 1e-5 * torch.eye(sigma0.shape[0], dtype=self.dtype))
            generator = torch.Generator(device=mu0.device)
            generator.manual_seed(seed)
            eps = torch.randn(self.n_particles, mu0.shape[0], dtype=self.dtype, generator=generator)
            base_particles = mu0.unsqueeze(0) + eps @ chol.t()
        def make_generator() -> torch.Generator:
            generator = torch.Generator(device=base_particles.device)
            generator.manual_seed(seed)
            return generator

        torch.manual_seed(seed)
        masks = self.net.sample_masks(base_particles.shape[0], device=base_particles.device)
        masks = [m.to(base_particles.dtype) for m in masks]

        def imagined_cost() -> float:
            with torch.no_grad():
                cost, _ = predict_trajectory_particles(
                    self.net,
                    self.policy,
                    particles0=base_particles,
                    masks=masks,
                    horizon=self.horizon,
                    target=self.target,
                    weight=self.weight,
                    action_high=self.action_high_t,
                    project_encoded_angle=self.encode_angle,
                    step_cost_fn=self._particle_step_cost_fn(),
                    generator=make_generator(),
                )
            return float(cost.item())

        before = imagined_cost()
        best_cost = before
        best_policy = [p.detach().clone() for p in self.policy.parameters()]

        for _ in range(self.policy_opt_steps):
            optimizer.zero_grad()
            cost, _ = predict_trajectory_particles(
                self.net,
                self.policy,
                particles0=base_particles,
                masks=masks,
                horizon=self.horizon,
                target=self.target,
                weight=self.weight,
                action_high=self.action_high_t,
                project_encoded_angle=self.encode_angle,
                step_cost_fn=self._particle_step_cost_fn(),
                generator=make_generator(),
            )
            if not torch.isfinite(cost):
                break
            cost.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
            optimizer.step()
            current = float(cost.detach().item())
            if math.isfinite(current) and current < best_cost:
                best_cost = current
                best_policy = [p.detach().clone() for p in self.policy.parameters()]

        with torch.no_grad():
            for param, saved in zip(self.policy.parameters(), best_policy):
                param.copy_(saved)
        after = imagined_cost()
        if (not math.isfinite(after)) or after > before:
            with torch.no_grad():
                for param, saved in zip(self.policy.parameters(), backup):
                    param.copy_(saved)
            after = imagined_cost()
        for p in self.net.parameters():
            p.requires_grad_(True)
        return before, after

    def learn_step(self, **kwargs: Any) -> dict[str, float]:
        iteration = int(kwargs.pop("iteration", 0))
        del kwargs
        model_loss, model_metrics = self._train_bnn(iteration=iteration)
        predicted_before, predicted_after = self._optimize_policy(iteration=iteration)
        return {
            "model_loss": float(model_loss) if math.isfinite(model_loss) else 0.0,
            "model_train_loss": float(model_metrics.get("train_loss", 0.0)),
            "model_val_loss": float(model_metrics.get("val_loss", 0.0)),
            "model_weight_decay": float(model_metrics.get("weight_decay", 0.0)),
            "model_grad_clip": float(model_metrics.get("grad_clip", 0.0)),
            "predicted_cost_before": float(predicted_before),
            "predicted_cost": float(predicted_after) if math.isfinite(predicted_after) else float(predicted_before),
            "gp_points": float(len(self.buffer)),
        }

    def episode_ended(self) -> None:
        return None

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "policy": self.policy.state_dict(),
                "buffer": self.buffer.state_dict(),
                "model_opt_state": self._model_opt_state,
                "target": self.target,
                "weight": self.weight,
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "DeepPilcoAgent":
        payload = torch.load(Path(path), weights_only=False)
        agent = cls(**kwargs)
        agent.net.load_state_dict(payload["net"])
        agent.policy.load_state_dict(payload["policy"])
        agent.buffer.load_state_dict(payload.get("buffer", {}))
        agent._model_opt_state = payload.get("model_opt_state")
        if "target" in payload:
            agent.target = payload["target"]
        if "weight" in payload:
            agent.weight = payload["weight"]
        return agent
