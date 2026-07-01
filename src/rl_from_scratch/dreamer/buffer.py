"""Sequence replay buffer for DreamerV1.

DreamerV1 needs contiguous subsequences of experiences (not i.i.d.
transitions) because the RSSM requires temporal context.
``SequenceBuffer`` stores complete episodes and samples random windows
of length ``seq_len`` from them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SequenceBuffer:
    """Episode-based buffer that samples contiguous sub-sequences.

    Parameters
    ----------
    capacity:
        Maximum number of *transitions* to keep (across all stored episodes).
        When the limit is exceeded, the oldest episodes are dropped.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._episodes: list[dict[str, np.ndarray]] = []
        self._current: list[tuple[Any, Any, float, bool]] = []

    # ------------------------------------------------------------------
    # Building episodes
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Append a single transition to the current in-progress episode.

        If ``done`` is True the episode is automatically flushed (sealed).
        """
        self._current.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                bool(done),
            )
        )
        if done:
            self._flush_and_enforce_capacity()

    def flush_current(self) -> None:
        """Seal the current in-progress episode (used on truncation)."""
        if self._current:
            self._flush_and_enforce_capacity()

    def _flush_and_enforce_capacity(self) -> None:
        """Move ``_current`` into ``_episodes`` and enforce capacity."""
        if not self._current:
            return
        ep = self._stack_current()
        self._episodes.append(ep)
        self._current = []
        # Drop oldest episodes until total transitions <= capacity
        while self._total_transitions() > self._capacity and len(self._episodes) > 1:
            self._episodes.pop(0)

    def _stack_current(self) -> dict[str, np.ndarray]:
        """Convert the current list of tuples into stacked arrays."""
        obss, actions, rewards, dones = zip(*self._current)
        return {
            "obs": np.stack(obss, axis=0).astype(np.float32),       # [L, obs_dim]
            "action": np.stack(actions, axis=0).astype(np.float32),  # [L, action_dim]
            "reward": np.array(rewards, dtype=np.float32),           # [L]
            "done": np.array(dones, dtype=np.float32),               # [L]
        }

    def _total_transitions(self) -> int:
        """Count transitions in sealed episodes only."""
        return sum(ep["obs"].shape[0] for ep in self._episodes)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        seq_len: int,
    ) -> dict[str, torch.Tensor]:
        """Sample ``batch_size`` contiguous windows of length ``seq_len``.

        Parameters
        ----------
        batch_size:
            Number of sub-sequences to return.
        seq_len:
            Length of each sub-sequence (``batch_length`` in config).

        Returns
        -------
        dict with float32 tensors:
            - ``obs``    ``[B, L, obs_dim]``
            - ``action`` ``[B, L, action_dim]``
            - ``reward`` ``[B, L]``
            - ``done``   ``[B, L]``

        Raises
        ------
        RuntimeError
            If no stored episode is long enough to yield a sub-sequence
            of the requested length.
        """
        eligible = [ep for ep in self._episodes if ep["obs"].shape[0] >= seq_len]
        if not eligible:
            raise RuntimeError(
                f"SequenceBuffer: no episode with length >= {seq_len}. "
                f"Stored episodes have lengths "
                f"{[ep['obs'].shape[0] for ep in self._episodes]}."
            )

        obs_list, act_list, rew_list, done_list = [], [], [], []
        for _ in range(batch_size):
            ep = eligible[np.random.randint(len(eligible))]
            L = ep["obs"].shape[0]
            start = np.random.randint(0, L - seq_len + 1)
            sl = slice(start, start + seq_len)
            obs_list.append(ep["obs"][sl])
            act_list.append(ep["action"][sl])
            rew_list.append(ep["reward"][sl])
            done_list.append(ep["done"][sl])

        return {
            "obs": torch.tensor(np.stack(obs_list), dtype=torch.float32),
            "action": torch.tensor(np.stack(act_list), dtype=torch.float32),
            "reward": torch.tensor(np.stack(rew_list), dtype=torch.float32),
            "done": torch.tensor(np.stack(done_list), dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total stored transitions (sealed + current in-progress)."""
        sealed = self._total_transitions()
        return sealed + len(self._current)
