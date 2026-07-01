"""Transition buffer for PILCO and Deep PILCO."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor


class TransitionBuffer:
    """FIFO buffer of transitions plus stable train/holdout splitting metadata."""

    def __init__(self, max_gp_points: int = 300, rng: np.random.Generator | None = None) -> None:
        self.max_gp_points = max_gp_points
        self.rng = rng or np.random.default_rng()
        self._obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._next_obs: list[np.ndarray] = []
        self._failure: list[bool] = []
        self._episode_initial_obs: list[np.ndarray] = []
        self._holdout_indices: list[int] | None = None

    def start_episode(self, obs: Any) -> None:
        self._episode_initial_obs.append(np.asarray(obs, dtype=np.float64))

    def push(self, obs: Any, action: Any, next_obs: Any, failure: bool = False) -> None:
        self._obs.append(np.asarray(obs, dtype=np.float64))
        self._actions.append(np.asarray(action, dtype=np.float64).flatten())
        self._next_obs.append(np.asarray(next_obs, dtype=np.float64))
        self._failure.append(bool(failure))

    def _stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self._obs:
            raise RuntimeError("Buffer is empty; cannot construct training tensors.")
        return (
            np.stack(self._obs),
            np.stack(self._actions),
            np.stack(self._next_obs),
            np.asarray(self._failure, dtype=bool),
        )

    def _stratified_choice(
        self,
        indices: np.ndarray,
        failure: np.ndarray,
        obs_arr: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count >= len(indices):
            return np.sort(indices.copy())
        if obs_arr.shape[1] >= 2:
            safe_mask = np.abs(obs_arr[:, 1]) <= np.sin(0.25)
            safe_idx = indices[safe_mask[indices]]
            outside_idx = indices[~safe_mask[indices]]
        else:
            outside_idx = indices[failure[indices]]
            safe_idx = indices[~failure[indices]]
        # Notebook recipe: 80% upright / 20% outside the local stabilisation tube.
        target_safe = min(len(safe_idx), int(0.8 * count))
        target_outside = min(len(outside_idx), count - target_safe)
        chosen: list[int] = []
        if target_safe:
            chosen.extend(int(x) for x in rng.choice(safe_idx, size=target_safe, replace=False))
        if target_outside:
            chosen.extend(int(x) for x in rng.choice(outside_idx, size=target_outside, replace=False))
        remaining = count - len(chosen)
        pool = np.setdiff1d(indices, np.asarray(chosen, dtype=int), assume_unique=False)
        if remaining > 0:
            extra = rng.choice(pool, size=remaining, replace=False)
            chosen.extend(int(x) for x in extra)
        return np.sort(np.asarray(chosen, dtype=int))

    def get_recent_tensors(
        self,
        *,
        cap: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        obs_arr, act_arr, nobs_arr, _ = self._stack()
        if cap is not None and cap > 0 and len(obs_arr) > cap:
            obs_arr = obs_arr[-cap:]
            act_arr = act_arr[-cap:]
            nobs_arr = nobs_arr[-cap:]
        x = torch.tensor(
            np.concatenate([obs_arr, act_arr], axis=1),
            dtype=torch.float64,
        )
        y = torch.tensor(nobs_arr - obs_arr, dtype=torch.float64)
        return x, y

    def get_train_and_holdout_tensors(
        self,
        *,
        validation_fraction: float = 0.15,
        validation_min_points: int = 32,
        seed: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        obs_arr, act_arr, nobs_arr, failure = self._stack()
        n = len(obs_arr)
        all_indices = np.arange(n, dtype=int)

        if self._holdout_indices is None:
            holdout_count = min(
                max(validation_min_points, int(math.ceil(validation_fraction * n))),
                max(n - 1, 0),
            )
            if holdout_count > 0:
                rng = np.random.default_rng(seed)
                self._holdout_indices = self._stratified_choice(
                    all_indices,
                    failure,
                    obs_arr,
                    holdout_count,
                    rng,
                ).tolist()
            else:
                self._holdout_indices = []

        holdout_idx = np.asarray(self._holdout_indices, dtype=int)
        train_idx = np.setdiff1d(all_indices, holdout_idx, assume_unique=False)
        if len(train_idx) > self.max_gp_points:
            rng = np.random.default_rng(seed + 1)
            train_idx = self._stratified_choice(train_idx, failure, obs_arr, self.max_gp_points, rng)

        def _to_tensors(indices: np.ndarray) -> tuple[Tensor, Tensor]:
            if len(indices) == 0:
                input_dim = obs_arr.shape[1] + act_arr.shape[1]
                output_dim = obs_arr.shape[1]
                return (
                    torch.zeros((0, input_dim), dtype=torch.float64),
                    torch.zeros((0, output_dim), dtype=torch.float64),
                )
            x = torch.tensor(
                np.concatenate([obs_arr[indices], act_arr[indices]], axis=1),
                dtype=torch.float64,
            )
            y = torch.tensor(nobs_arr[indices] - obs_arr[indices], dtype=torch.float64)
            return x, y

        train_x, train_y = _to_tensors(train_idx)
        holdout_x, holdout_y = _to_tensors(holdout_idx)
        meta = {
            "train_indices": train_idx.tolist(),
            "holdout_indices": holdout_idx.tolist(),
            "train_failure_count": int(failure[train_idx].sum()) if len(train_idx) else 0,
            "train_safe_count": int((~failure[train_idx]).sum()) if len(train_idx) else 0,
        }
        return train_x, train_y, holdout_x, holdout_y, meta

    def get_gp_tensors(self, seed: int | None = None) -> tuple[Tensor, Tensor]:
        train_x, train_y, _, _, _ = self.get_train_and_holdout_tensors(seed=0 if seed is None else seed)
        return train_x, train_y

    def initial_state_mean(self) -> np.ndarray:
        if not self._obs:
            raise RuntimeError("Buffer is empty.")
        if self._episode_initial_obs:
            return np.mean(self._episode_initial_obs, axis=0)
        return self._obs[0].copy()

    def initial_state_cov(self) -> np.ndarray:
        if len(self._episode_initial_obs) < 2:
            if self._obs:
                d = self._obs[0].shape[0]
            else:
                raise RuntimeError("Buffer is empty.")
            return np.full(d, 1e-4)
        arr = np.stack(self._episode_initial_obs)
        return np.var(arr, axis=0).clip(1e-6)

    def initial_state_samples(self, count: int) -> np.ndarray:
        """Return deterministic reset representatives for multi-belief planning."""
        if count <= 0:
            raise ValueError("count must be positive.")
        if not self._episode_initial_obs:
            if not self._obs:
                raise RuntimeError("Buffer is empty.")
            return np.repeat(self._obs[0][None, :], count, axis=0)
        starts = np.stack(self._episode_initial_obs)
        if len(starts) >= count:
            indices = np.linspace(0, len(starts) - 1, num=count, dtype=int)
            return starts[indices].copy()
        repeats = int(np.ceil(count / len(starts)))
        return np.tile(starts, (repeats, 1))[:count].copy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs": list(self._obs),
            "actions": list(self._actions),
            "next_obs": list(self._next_obs),
            "failure": list(self._failure),
            "episode_initial_obs": list(self._episode_initial_obs),
            "holdout_indices": list(self._holdout_indices or []),
            "max_gp_points": self.max_gp_points,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._obs = [np.asarray(o, dtype=np.float64) for o in payload.get("obs", [])]
        self._actions = [np.asarray(a, dtype=np.float64) for a in payload.get("actions", [])]
        self._next_obs = [np.asarray(n, dtype=np.float64) for n in payload.get("next_obs", [])]
        self._failure = [bool(v) for v in payload.get("failure", [])]
        self._episode_initial_obs = [
            np.asarray(o, dtype=np.float64) for o in payload.get("episode_initial_obs", [])
        ]
        self._holdout_indices = [int(i) for i in payload.get("holdout_indices", [])]
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def __len__(self) -> int:
        return len(self._obs)
