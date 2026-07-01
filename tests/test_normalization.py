"""Tests pour le module normalization (RunningMeanStd, ObservationNormalizer)."""

from __future__ import annotations

import numpy as np
import pytest

from rl_from_scratch.core.normalization import ObservationNormalizer, RunningMeanStd


# ------------------------------------------------------------------
# RunningMeanStd
# ------------------------------------------------------------------


def test_running_mean_std_initial_values() -> None:
    """À l'initialisation, mean=0 et var=1 pour toute shape."""
    rms = RunningMeanStd(shape=(4,))
    np.testing.assert_array_equal(rms.mean, np.zeros(4))
    np.testing.assert_array_equal(rms.var, np.ones(4))
    assert rms.count == 0.0


def test_running_mean_std_single_update() -> None:
    """Après un batch d'une seule observation, mean et var correspondent au batch."""
    rms = RunningMeanStd(shape=(3,))
    obs = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    rms.update(obs)

    np.testing.assert_allclose(rms.mean, np.array([1.0, 2.0, 3.0]), atol=1e-10)
    # var d'un seul élément → 0
    np.testing.assert_allclose(rms.var, np.zeros(3), atol=1e-10)
    assert rms.count == 1.0


def test_running_mean_std_batch_update() -> None:
    """Après plusieurs updates, mean et var convergent vers les vraies statistiques."""
    rng = np.random.default_rng(42)
    obs_dim = 4
    n_samples = 1000

    data = rng.standard_normal((n_samples, obs_dim)).astype(np.float64)
    true_mean = data.mean(axis=0)
    true_var = data.var(axis=0)

    rms = RunningMeanStd(shape=(obs_dim,))
    # Met à jour par mini-batchs pour tester l'accumulation incrémentale
    batch_size = 50
    for i in range(0, n_samples, batch_size):
        rms.update(data[i : i + batch_size])

    np.testing.assert_allclose(rms.mean, true_mean, atol=1e-10)
    np.testing.assert_allclose(rms.var, true_var, atol=1e-10)
    assert rms.count == float(n_samples)


def test_running_mean_std_serialization() -> None:
    """to_dict/from_dict préserve mean, var et count."""
    rng = np.random.default_rng(7)
    rms = RunningMeanStd(shape=(4,))
    rms.update(rng.standard_normal((20, 4)))

    d = rms.to_dict()
    rms2 = RunningMeanStd.from_dict(d)

    np.testing.assert_array_equal(rms2.mean, rms.mean)
    np.testing.assert_array_equal(rms2.var, rms.var)
    assert rms2.count == rms.count


# ------------------------------------------------------------------
# ObservationNormalizer
# ------------------------------------------------------------------


def test_normalizer_normalize_updates_stats() -> None:
    """normalize(obs, update=True) met à jour les statistiques internes."""
    normalizer = ObservationNormalizer(obs_dim=4, epsilon=1e-8, clip=10.0)
    assert normalizer.rms.count == 0.0

    obs = np.ones(4, dtype=np.float32)
    normalizer.normalize(obs, update=True)

    assert normalizer.rms.count == 1.0
    np.testing.assert_allclose(normalizer.rms.mean, np.ones(4), atol=1e-6)


def test_normalizer_normalize_frozen() -> None:
    """normalize(obs, update=False) ne modifie pas les statistiques internes."""
    normalizer = ObservationNormalizer(obs_dim=4, epsilon=1e-8, clip=10.0)

    # Première passe pour initialiser les stats
    obs1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    normalizer.normalize(obs1, update=True)
    count_after_train = normalizer.rms.count
    mean_after_train = normalizer.rms.mean.copy()

    # Deuxième passe en mode évaluation — les stats ne doivent pas changer
    obs2 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    normalizer.normalize(obs2, update=False)

    assert normalizer.rms.count == count_after_train
    np.testing.assert_array_equal(normalizer.rms.mean, mean_after_train)


def test_normalizer_clip() -> None:
    """La sortie normalisée est contenue dans [-clip, clip]."""
    clip = 5.0
    normalizer = ObservationNormalizer(obs_dim=4, epsilon=1e-8, clip=clip)

    # Met d'abord à jour les stats avec des valeurs modérées
    rng = np.random.default_rng(0)
    for _ in range(50):
        normalizer.normalize(rng.standard_normal(4).astype(np.float32), update=True)

    # Envoie une observation extrême
    extreme_obs = np.array([1000.0, -1000.0, 500.0, -500.0], dtype=np.float32)
    normalized = normalizer.normalize(extreme_obs, update=False)

    assert (normalized >= -clip).all(), f"Valeur en dessous de -{clip}: {normalized.min()}"
    assert (normalized <= clip).all(), f"Valeur au dessus de {clip}: {normalized.max()}"


def test_normalizer_serialization() -> None:
    """to_dict/from_dict préserve toutes les statistiques et hyperparamètres."""
    normalizer = ObservationNormalizer(obs_dim=4, epsilon=1e-6, clip=5.0)

    rng = np.random.default_rng(99)
    for _ in range(30):
        normalizer.normalize(rng.standard_normal(4).astype(np.float32), update=True)

    d = normalizer.to_dict()
    normalizer2 = ObservationNormalizer.from_dict(d)

    assert normalizer2.obs_dim == normalizer.obs_dim
    assert normalizer2.epsilon == normalizer.epsilon
    assert normalizer2.clip == normalizer.clip
    np.testing.assert_array_equal(normalizer2.rms.mean, normalizer.rms.mean)
    np.testing.assert_array_equal(normalizer2.rms.var, normalizer.rms.var)
    assert normalizer2.rms.count == normalizer.rms.count

    # La même observation doit produire exactement le même output
    obs = rng.standard_normal(4).astype(np.float32)
    out1 = normalizer.normalize(obs, update=False)
    out2 = normalizer2.normalize(obs, update=False)
    np.testing.assert_array_equal(out1, out2)


def test_normalizer_batch_and_single() -> None:
    """normalize fonctionne avec une obs de forme (obs_dim,) et un batch (N, obs_dim)."""
    obs_dim = 4
    normalizer_single = ObservationNormalizer(obs_dim=obs_dim)
    normalizer_batch = ObservationNormalizer(obs_dim=obs_dim)

    rng = np.random.default_rng(11)
    data = rng.standard_normal((10, obs_dim)).astype(np.float32)

    # Met à jour les deux normaliseurs avec les mêmes données
    for row in data:
        normalizer_single.normalize(row, update=True)
    normalizer_batch.normalize(data, update=True)

    # Les stats doivent être identiques
    np.testing.assert_allclose(
        normalizer_single.rms.mean, normalizer_batch.rms.mean, atol=1e-10
    )
    np.testing.assert_allclose(
        normalizer_single.rms.var, normalizer_batch.rms.var, atol=1e-10
    )
    assert normalizer_single.rms.count == normalizer_batch.rms.count

    # Inférence sur une obs unique
    obs = rng.standard_normal(obs_dim).astype(np.float32)
    out_single = normalizer_single.normalize(obs, update=False)
    assert out_single.shape == (obs_dim,), f"Attendu ({obs_dim},), obtenu {out_single.shape}"

    # Inférence sur un batch
    batch = rng.standard_normal((5, obs_dim)).astype(np.float32)
    out_batch = normalizer_batch.normalize(batch, update=False)
    assert out_batch.shape == (5, obs_dim), f"Attendu (5, {obs_dim}), obtenu {out_batch.shape}"


def test_normalizer_output_dtype() -> None:
    """La sortie de normalize est toujours float32."""
    normalizer = ObservationNormalizer(obs_dim=4)
    obs = np.ones(4, dtype=np.float64)
    out = normalizer.normalize(obs, update=True)
    assert out.dtype == np.float32, f"Attendu float32, obtenu {out.dtype}"
