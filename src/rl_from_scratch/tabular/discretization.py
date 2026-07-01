"""State discretization helpers for classic control environments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class CartPoleDiscretizer:
    """Map continuous CartPole observations to tabular state indices."""

    def __init__(
        self,
        bins: Iterable[int] | Iterable[np.ndarray] | None = None,
        *,
        config: object | None = None,
        bin_counts: Iterable[int] | None = None,
        num_bins: Iterable[int] | None = None,
        low: Iterable[float] | None = None,
        high: Iterable[float] | None = None,
        lower_bounds: Iterable[float] | None = None,
        upper_bounds: Iterable[float] | None = None,
    ) -> None:
        if bins is not None:
            raw_bins = tuple(bins)
        elif bin_counts is not None:
            raw_bins = tuple(bin_counts)
        elif num_bins is not None:
            raw_bins = tuple(num_bins)
        elif config is not None:
            raw_bins = config.bins  # type: ignore[union-attr]
        else:
            # Lazy import to avoid circular dependency
            from rl_from_scratch.tabular.config import QLearningConfig

            raw_bins = QLearningConfig().bins

        if raw_bins and isinstance(raw_bins[0], np.ndarray):
            self.bins = tuple(np.asarray(edges, dtype=np.float64) for edges in raw_bins)  # type: ignore[assignment]
            return

        counts = tuple(int(count) for count in raw_bins)
        if len(counts) != 4:
            raise ValueError("CartPole discretization requires four bin counts.")

        if config is None:
            from rl_from_scratch.tabular.config import QLearningConfig

            config = QLearningConfig(bins=counts)

        lower = np.asarray(
            lower_bounds
            if lower_bounds is not None
            else low
            if low is not None
            else (
                -4.8,
                config.cart_velocity_min,  # type: ignore[union-attr]
                -0.418,
                config.pole_angular_velocity_min,  # type: ignore[union-attr]
            ),
            dtype=np.float64,
        )
        upper = np.asarray(
            upper_bounds
            if upper_bounds is not None
            else high
            if high is not None
            else (
                4.8,
                config.cart_velocity_max,  # type: ignore[union-attr]
                0.418,
                config.pole_angular_velocity_max,  # type: ignore[union-attr]
            ),
            dtype=np.float64,
        )
        self.bins = tuple(
            np.linspace(lower[index], upper[index], counts[index])
            for index in range(4)
        )  # type: ignore[assignment]

    @classmethod
    def from_config(
        cls, config: object, observation_space: object
    ) -> CartPoleDiscretizer:
        high = np.asarray(observation_space.high, dtype=np.float64).copy()  # type: ignore[union-attr]
        low = np.asarray(observation_space.low, dtype=np.float64).copy()  # type: ignore[union-attr]

        low[1] = config.cart_velocity_min  # type: ignore[union-attr]
        high[1] = config.cart_velocity_max  # type: ignore[union-attr]
        low[3] = config.pole_angular_velocity_min  # type: ignore[union-attr]
        high[3] = config.pole_angular_velocity_max  # type: ignore[union-attr]

        edges = tuple(
            np.linspace(low[index], high[index], config.bins[index])  # type: ignore[union-attr]
            for index in range(4)
        )
        return cls(edges)

    def discretize(self, observation: Iterable[float]) -> tuple[int, int, int, int]:
        return self.transform(observation)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(len(edges) for edges in self.bins)  # type: ignore[return-value]

    def transform(self, observation: Iterable[float]) -> tuple[int, int, int, int]:
        values = np.asarray(tuple(observation), dtype=np.float64)
        if values.shape != (4,):
            raise ValueError(
                f"Expected a CartPole observation with shape (4,), got {values.shape}."
            )

        indices: list[int] = []
        for value, edges in zip(values, self.bins):
            index = int(np.digitize(value, edges) - 1)
            indices.append(int(np.clip(index, 0, len(edges) - 1)))
        return tuple(indices)  # type: ignore[return-value]
