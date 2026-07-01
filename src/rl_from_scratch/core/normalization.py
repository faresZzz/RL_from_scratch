"""Observation normalization via running means/variances (Welford).

Provides ``RunningMeanStd`` (online Welford algorithm) and
``ObservationNormalizer`` (wrapper with clipping) to stabilize on-policy
training on continuous environments such as MuJoCo.

Minimal usage ::

    normalizer = ObservationNormalizer(obs_dim=17)
    obs_norm = normalizer.normalize(obs, update=True)   # training
    obs_norm = normalizer.normalize(obs, update=False)  # evaluation
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RunningMeanStd:
    """Online mean and variance via the Welford algorithm.

    Thread-safety not required (single-process only).

    Parameters
    ----------
    shape:
        Shape of the statistic (e.g. ``(obs_dim,)`` for a vector,
        ``()`` for a scalar).
    """

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.var: np.ndarray = np.ones(shape, dtype=np.float64)
        self.count: float = 0.0

    def update(self, batch: np.ndarray) -> None:
        """Update the statistics with a batch of observations.

        Uses the batched generalization of the Welford update formula to
        avoid two passes over the data.

        Parameters
        ----------
        batch:
            Array of shape ``(N, *shape)`` or ``shape`` (single observation).
        """
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            # Single observation — add a batch dimension
            batch = batch[np.newaxis]

        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize the running statistics to a Python dict."""
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunningMeanStd:
        """Rebuild a ``RunningMeanStd`` from a serialized dict.

        Parameters
        ----------
        data:
            Dict produced by :meth:`to_dict`.

        Returns
        -------
        RunningMeanStd
            Instance with the statistics restored.
        """
        mean = np.array(data["mean"], dtype=np.float64)
        instance = cls(shape=mean.shape)
        instance.mean = mean
        instance.var = np.array(data["var"], dtype=np.float64)
        instance.count = float(data["count"])
        return instance


class ObservationNormalizer:
    """Normalize observations by (obs - mean) / sqrt(var + ε), clipped.

    Maintains running statistics via ``RunningMeanStd``.  In training mode
    (``update=True``) the statistics are updated before normalization.  In
    evaluation mode (``update=False``) the statistics are frozen.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    epsilon:
        Value added to the variance to avoid division by zero.
    clip:
        Maximum absolute value of the normalized observation.
    """

    def __init__(
        self,
        obs_dim: int,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.clip = clip
        self.rms = RunningMeanStd(shape=(obs_dim,))

    def normalize(
        self,
        obs: np.ndarray,
        update: bool = True,
    ) -> np.ndarray:
        """Normalize an observation or a batch of observations.

        Parameters
        ----------
        obs:
            Observation of shape ``(obs_dim,)`` or batch ``(N, obs_dim)``.
        update:
            If ``True``, update the running statistics before normalizing
            (training mode).  If ``False``, the statistics are frozen
            (evaluation mode).

        Returns
        -------
        np.ndarray
            Normalized observation of the same shape as the input, clipped to
            ``[-clip, clip]``, in ``float32``.
        """
        obs = np.asarray(obs, dtype=np.float32)
        if update:
            self.rms.update(obs)
        mean = self.rms.mean.astype(np.float32)
        std = np.sqrt(self.rms.var.astype(np.float32) + self.epsilon)
        normalized = (obs - mean) / std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the normalizer (statistics + hyperparameters)."""
        return {
            "obs_dim": self.obs_dim,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "rms": self.rms.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationNormalizer:
        """Rebuild an ``ObservationNormalizer`` from a serialized dict.

        Parameters
        ----------
        data:
            Dict produced by :meth:`to_dict`.

        Returns
        -------
        ObservationNormalizer
            Instance with the statistics and hyperparameters restored.
        """
        instance = cls(
            obs_dim=int(data["obs_dim"]),
            epsilon=float(data["epsilon"]),
            clip=float(data["clip"]),
        )
        instance.rms = RunningMeanStd.from_dict(data["rms"])
        return instance
