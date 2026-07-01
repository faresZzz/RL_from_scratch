"""Shared utilities for device resolution, seeding, and math helpers."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import nn


def resolve_device(preference: str = "auto") -> str:
    """Return the best available compute device as a string.

    Parameters
    ----------
    preference:
        ``"auto"`` picks MPS/CUDA/CPU in that order.  An explicit value
        (``"cpu"``, ``"cuda"``, ``"mps"``) is returned as-is.

    Torch is imported lazily so tabular-only code never requires it.
    """
    if preference != "auto":
        return preference
    try:
        import torch  # noqa: WPS433 (lazy import by design)

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def set_all_seeds(seed: int, env: object | None = None) -> None:
    """Seed Python, NumPy, and (optionally) Torch and a Gymnasium env.

    Parameters
    ----------
    seed:
        The integer seed to use everywhere.
    env:
        If provided, ``env.reset(seed=seed)`` and
        ``env.action_space.seed(seed)`` are called.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    if env is not None:
        env.reset(seed=seed)  # type: ignore[union-attr]
        env.action_space.seed(seed)  # type: ignore[union-attr]


def moving_average(values: list[float], window: int = 20) -> float:
    """Return the mean of the last *window* elements in *values*.

    Returns ``0.0`` for an empty list.
    """
    if not values:
        return 0.0
    return float(np.mean(values[-window:]))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average *source* into *target* in place.

    Updates ``target ← (1 - tau)·target + tau·source`` parameter by parameter.
    A small *tau* (e.g. ``0.005``) keeps the target slow-moving — the standard
    DDPG/SAC convention. For an exponential-moving-average target where *tau* is
    instead the weight kept on the slow target (e.g. ``0.99``), pass ``1 - tau``.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
