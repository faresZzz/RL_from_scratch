"""Transition buffer for PETS.

PETS uses all observed transitions to fit the probabilistic ensemble at
the start of each episode.  This buffer stores (obs, action, next_obs)
triples in float32 and provides helpers to assemble the ensemble training
tensors:

    X = [obs, action]           shape [N, obs_dim + act_dim]
    Y = next_obs - obs          shape [N, obs_dim]   (delta dynamics)

The buffer grows without bound — PETS trains on the full dataset every
iteration, so the natural representation is a growing list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


class TransitionBuffer:
    """Growing buffer of (obs, action, next_obs) triples in float32.

    Parameters
    ----------
    obs_dim:
        Observation dimensionality (used for pre-allocation hints only).
    act_dim:
        Action dimensionality (used for pre-allocation hints only).
    """

    def __init__(self, obs_dim: int = 0, act_dim: int = 0) -> None:
        del obs_dim, act_dim  # reserved for future pre-allocation
        self._obs: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._next_obs: list[np.ndarray] = []

    def push(self, obs: Any, action: Any, next_obs: Any) -> None:
        """Store a single (obs, action, next_obs) transition."""
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32).flatten())
        self._next_obs.append(np.asarray(next_obs, dtype=np.float32))

    def get_tensors(self) -> tuple[Tensor, Tensor]:
        """Return ensemble training tensors ``(X, Y)``.

        X : [N, obs_dim + act_dim]  — concatenated observation and action
        Y : [N, obs_dim]            — state delta  (next_obs - obs)

        Raises
        ------
        RuntimeError
            If the buffer is empty.
        """
        if not self._obs:
            raise RuntimeError("Buffer is empty; cannot construct training tensors.")

        obs_arr = np.stack(self._obs)        # [N, obs_dim]
        act_arr = np.stack(self._actions)    # [N, act_dim]
        nobs_arr = np.stack(self._next_obs)  # [N, obs_dim]

        X = torch.tensor(
            np.concatenate([obs_arr, act_arr], axis=1), dtype=torch.float32
        )
        Y = torch.tensor(nobs_arr - obs_arr, dtype=torch.float32)
        return X, Y

    def state_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the buffer contents."""
        return {
            "obs": [o.tolist() for o in self._obs],
            "actions": [a.tolist() for a in self._actions],
            "next_obs": [n.tolist() for n in self._next_obs],
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore buffer contents from a previously saved snapshot."""
        self._obs = [
            np.asarray(o, dtype=np.float32) for o in payload.get("obs", [])
        ]
        self._actions = [
            np.asarray(a, dtype=np.float32) for a in payload.get("actions", [])
        ]
        self._next_obs = [
            np.asarray(n, dtype=np.float32) for n in payload.get("next_obs", [])
        ]

    def __len__(self) -> int:
        return len(self._obs)
