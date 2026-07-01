"""Tests for utils from rl_from_scratch.core.utils."""

import numpy as np
import pytest
import torch
from torch import nn

from rl_from_scratch.core.utils import (
    moving_average,
    resolve_device,
    set_all_seeds,
    soft_update,
)


def test_resolve_device_returns_valid_string():
    device = resolve_device("auto")
    assert device in {"cpu", "cuda", "mps"}


def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu") == "cpu"


def test_set_all_seeds_makes_numpy_reproducible():
    set_all_seeds(123)
    a = np.random.rand(5)
    set_all_seeds(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_moving_average_basic():
    result = moving_average([1, 2, 3, 4, 5], window=3)
    assert result == pytest.approx(4.0)


def test_moving_average_empty():
    result = moving_average([], window=5)
    assert result == pytest.approx(0.0)


def test_soft_update_matches_ddpg_sac_formula():
    """soft_update(target, source, tau) == old θ_target ← τ·θ_online + (1-τ)·θ_target."""
    torch.manual_seed(0)
    target = nn.Linear(4, 3)
    source = nn.Linear(4, 3)
    tau = 0.1
    expected = [
        tau * s.data.clone() + (1.0 - tau) * t.data.clone()
        for t, s in zip(target.parameters(), source.parameters())
    ]
    soft_update(target, source, tau)
    for got, want in zip(target.parameters(), expected):
        assert torch.allclose(got.data, want)


def test_soft_update_reproduces_ema_with_complement_tau():
    """EMA θ_target ← τ·θ_target + (1-τ)·θ_online via soft_update(target, online, 1-τ)."""
    torch.manual_seed(0)
    target = nn.Linear(4, 3)
    online = nn.Linear(4, 3)
    ema_tau = 0.99
    expected = [
        ema_tau * t.data.clone() + (1.0 - ema_tau) * o.data.clone()
        for t, o in zip(target.parameters(), online.parameters())
    ]
    soft_update(target, online, 1.0 - ema_tau)
    for got, want in zip(target.parameters(), expected):
        assert torch.allclose(got.data, want)
