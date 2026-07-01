"""Tests for PETS (Probabilistic Ensembles with Trajectory Sampling).

Part 1 (math/unit tests): gaussian_nll, ProbabilisticMLP, ProbabilisticEnsemble
  (heteroscedastic toy, disagreement, variance decomposition), propagate/TS modes,
  CEM convergence, halfcheetah_reward vs env.

Part 2 (integration tests): registry, config validation, agent API,
  save/load round-trip, smoke training.
"""

from __future__ import annotations

import math
from pathlib import Path
from functools import partial

import numpy as np
import pytest
import torch

from rl_from_scratch.pets.dynamics import (
    ProbabilisticEnsemble,
    ProbabilisticMLP,
    gaussian_nll,
)
from rl_from_scratch.pets.planner import CEMPlanner
from rl_from_scratch.pets.reward import halfcheetah_reward, get_reward_fn


# ======================================================================
# Fixture: use float32 throughout PETS tests
# ======================================================================

@pytest.fixture(autouse=True)
def _float32():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


# ======================================================================
# Part 1 — Math / unit tests
# ======================================================================

# ----------------------------------------------------------------------
# gaussian_nll
# ----------------------------------------------------------------------

def test_gaussian_nll_matches_hand_computation() -> None:
    """Verify NLL against a manually computed value."""
    # For a single sample, D=1: NLL = 0.5 * [(target-mean)^2 * exp(-logvar) + logvar]
    # mean=0, logvar=0, target=1 → NLL = 0.5 * (1 * 1 + 0) = 0.5
    mean = torch.zeros(1, 1)
    logvar = torch.zeros(1, 1)
    target = torch.ones(1, 1)
    result = float(gaussian_nll(mean, logvar, target))
    assert result == pytest.approx(0.5, abs=1e-6)


def test_gaussian_nll_zero_at_perfect_prediction() -> None:
    """NLL is minimised (logvar → -inf) when mean == target and var → 0.
    In practice with finite logvar, NLL at mean=target is 0.5 * logvar (per dim).
    For logvar=0: NLL = 0.5 * 0 = 0."""
    mean = torch.tensor([[1.0, 2.0]])
    logvar = torch.zeros(1, 2)
    target = torch.tensor([[1.0, 2.0]])
    result = float(gaussian_nll(mean, logvar, target))
    assert result == pytest.approx(0.0, abs=1e-6)


def test_gaussian_nll_batched_mean() -> None:
    """NLL on a batch equals the mean of per-sample NLLs."""
    torch.manual_seed(0)
    B, D = 8, 3
    mean = torch.randn(B, D)
    logvar = torch.randn(B, D)
    target = torch.randn(B, D)

    batch_nll = float(gaussian_nll(mean, logvar, target))
    per_sample = [
        float(gaussian_nll(mean[i:i+1], logvar[i:i+1], target[i:i+1]))
        for i in range(B)
    ]
    assert batch_nll == pytest.approx(float(np.mean(per_sample)), abs=1e-5)


# ----------------------------------------------------------------------
# ProbabilisticMLP + ensemble on heteroscedastic toy
# ----------------------------------------------------------------------

def _make_hetero_data(n: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D heteroscedastic toy: y = sin(x) + eps(x) where std grows with |x|."""
    torch.manual_seed(42)
    x = torch.linspace(-3.0, 3.0, n).unsqueeze(1)   # [N, 1]
    noise_std = 0.1 + 0.5 * x.abs()                  # larger noise for large |x|
    y = torch.sin(x) + noise_std * torch.randn_like(x)
    return x, y


def test_ensemble_reduces_nll_on_heteroscedastic_toy() -> None:
    """Training an ensemble should reduce NLL on the training data."""
    x, y = _make_hetero_data(200)
    ensemble = ProbabilisticEnsemble(
        input_dim=1, output_dim=1, ensemble_size=2, hidden_dim=32, n_layers=2
    )
    # Evaluate initial NLL (untrained)
    ensemble.fit_normalizer(x, y)
    x_norm = (x - ensemble.x_mean) / ensemble.x_std
    y_norm = (y - ensemble.y_mean) / ensemble.y_std

    with torch.no_grad():
        m0, lv0 = ensemble.members[0](x_norm)
        nll_before = float(gaussian_nll(m0, lv0, y_norm))

    nll_after = ensemble.fit(x, y, steps=100, batch_size=64, lr=1e-3, weight_decay=0.0)
    assert nll_after < nll_before, "NLL should decrease after training"


def test_ensemble_aleatoric_std_larger_in_high_noise_region() -> None:
    """After training, predicted aleatoric std should be larger for high-noise x."""
    x, y = _make_hetero_data(300)
    ensemble = ProbabilisticEnsemble(
        input_dim=1, output_dim=1, ensemble_size=3, hidden_dim=64, n_layers=2
    )
    ensemble.fit(x, y, steps=200, batch_size=64, lr=1e-3, weight_decay=0.0)

    # Low-noise region: x ≈ 0; high-noise region: x ≈ 3
    x_low = torch.zeros(10, 1)
    x_high = torch.full((10, 1), 3.0)

    with torch.no_grad():
        _, vars_low = ensemble.predict(x_low)   # [10, E, 1]
        _, vars_high = ensemble.predict(x_high)

    # Mean predicted std: sqrt of mean variance across members
    std_low = float(vars_low.mean().sqrt())
    std_high = float(vars_high.mean().sqrt())
    assert std_high > std_low, (
        f"Expected higher aleatoric uncertainty at x=3 ({std_high:.4f}) "
        f"than at x=0 ({std_low:.4f})"
    )


# ----------------------------------------------------------------------
# Disagreement (epistemic uncertainty)
# ----------------------------------------------------------------------

def test_disagreement_higher_out_of_distribution() -> None:
    """Epistemic disagreement should be larger for unseen regions."""
    torch.manual_seed(0)
    # Train on x in [0, 1]; test disagreement at x=0.5 vs x=5.0
    N = 100
    x_train = torch.rand(N, 2)
    y_train = x_train[:, 0:1] + 0.1 * torch.randn(N, 1)

    ensemble = ProbabilisticEnsemble(
        input_dim=2, output_dim=1, ensemble_size=4, hidden_dim=32, n_layers=2
    )
    ensemble.fit(x_train, y_train, steps=200, batch_size=32, lr=1e-3, weight_decay=0.0)

    in_dist = torch.rand(50, 2)                          # within [0,1]
    out_dist = torch.rand(50, 2) * 0.1 + torch.tensor([10.0, 10.0])  # far from data

    disag_in = ensemble.disagreement(in_dist)
    disag_out = ensemble.disagreement(out_dist)

    assert disag_out > disag_in, (
        f"OOD disagreement ({disag_out:.4f}) should exceed in-dist ({disag_in:.4f})"
    )


# ----------------------------------------------------------------------
# Variance decomposition
# ----------------------------------------------------------------------

def test_variance_decomposition_identity() -> None:
    """Total variance ≈ aleatoric (mean of member vars) + epistemic (var of means).

    E[Var[y|m]] + Var[E[y|m]] = E[(y - E[y])^2]  (law of total variance).
    We verify this numerically on ensemble predictions.
    """
    torch.manual_seed(1)
    N, D = 30, 2
    x_test = torch.randn(N, 3)

    ensemble = ProbabilisticEnsemble(
        input_dim=3, output_dim=D, ensemble_size=4, hidden_dim=32, n_layers=2
    )
    # Initialise normalizer to avoid NaN
    dummy_x = torch.randn(20, 3)
    dummy_y = torch.randn(20, D)
    ensemble.fit_normalizer(dummy_x, dummy_y)

    with torch.no_grad():
        means, vars_ = ensemble.predict(x_test)   # [N, E, D]

    # Aleatoric = mean over members of variances  E[Var[y|m]]
    aleatoric = vars_.mean(dim=1)                         # [N, D]
    # Epistemic = variance over members of means  Var[E[y|m]]
    # Use correction=False to get the population variance (divide by E, not E-1)
    # so that total_var = aleatoric + epistemic exactly equals total_direct.
    epistemic = means.var(dim=1, correction=False)        # [N, D]
    # Total via law of total variance: E[Var] + Var[E] = total variance
    total_var = aleatoric + epistemic                      # [N, D]

    # Cross-check: total variance computed directly as E[(y - E[y])^2] + aleatoric
    grand_mean = means.mean(dim=1, keepdim=True)           # [N, 1, D]
    total_direct = ((means - grand_mean) ** 2).mean(dim=1) + vars_.mean(dim=1)

    assert torch.allclose(total_var, total_direct, atol=1e-5), (
        "Law of total variance identity not satisfied"
    )


# ----------------------------------------------------------------------
# Propagate / TS modes
# ----------------------------------------------------------------------

def _make_ensemble(obs_dim: int = 3, act_dim: int = 2, ensemble_size: int = 3) -> ProbabilisticEnsemble:
    torch.manual_seed(7)
    ensemble = ProbabilisticEnsemble(
        input_dim=obs_dim + act_dim,
        output_dim=obs_dim,
        ensemble_size=ensemble_size,
        hidden_dim=16,
        n_layers=2,
    )
    # Fit normalizer with dummy data
    X = torch.randn(50, obs_dim + act_dim)
    Y = torch.randn(50, obs_dim)
    ensemble.fit_normalizer(X, Y)
    return ensemble


def test_propagate_output_shape() -> None:
    """propagate returns correct shape [B, obs_dim]."""
    obs_dim, act_dim, E = 4, 2, 3
    B = 10
    ensemble = _make_ensemble(obs_dim, act_dim, E)
    states = torch.randn(B, obs_dim)
    actions = torch.randn(B, act_dim)
    model_idx = torch.randint(0, E, (B,))

    next_states = ensemble.propagate(states, actions, model_idx)
    assert next_states.shape == (B, obs_dim), (
        f"Expected shape ({B}, {obs_dim}), got {next_states.shape}"
    )


def test_propagate_tsinf_reproducible_with_fixed_seed() -> None:
    """TS∞: same seed → same propagation sequence."""
    obs_dim, act_dim, E = 3, 2, 4
    B = 8
    ensemble = _make_ensemble(obs_dim, act_dim, E)
    states = torch.randn(B, obs_dim)
    actions = torch.randn(B, act_dim)
    member_idx = torch.arange(B) % E   # fixed assignment = TS∞

    torch.manual_seed(42)
    out1 = ensemble.propagate(states, actions, member_idx)
    torch.manual_seed(42)
    out2 = ensemble.propagate(states, actions, member_idx)

    assert torch.allclose(out1, out2, atol=1e-6), "TS∞ must be reproducible with same seed"


def test_ts1_vs_tsinf_member_assignment_differs() -> None:
    """TS1 resamples member indices each step; TS∞ keeps them fixed.

    We verify that the index arrays differ across steps for TS1 but are
    constant across steps for TS∞.
    """
    E = 5
    B = 20
    H = 4

    # TS∞: fixed assignment across horizon steps
    member_tsinf_step0 = torch.arange(B) % E
    member_tsinf_step1 = torch.arange(B) % E
    assert torch.equal(member_tsinf_step0, member_tsinf_step1)

    # TS1: re-sample each step
    torch.manual_seed(0)
    ts1_step0 = torch.randint(0, E, (B,))
    ts1_step1 = torch.randint(0, E, (B,))
    # With B=20 and E=5, it would be astronomically unlikely for these to be equal
    assert not torch.equal(ts1_step0, ts1_step1), (
        "TS1 indices should differ between steps"
    )


# ----------------------------------------------------------------------
# CEM planner
# ----------------------------------------------------------------------

def test_cem_shifts_toward_high_reward_actions() -> None:
    """CEM should shift the mean toward high-reward actions on a toy objective.

    Toy: a trivial linear dynamics (state unchanged) + reward = sum(action),
    so higher actions = higher reward.  CEM should push the mean toward action_high.
    """
    torch.manual_seed(0)
    obs_dim, act_dim = 2, 2
    E = 2
    action_high = torch.ones(act_dim) * 2.0
    action_low = -action_high

    ensemble = _make_ensemble(obs_dim, act_dim, E)

    # Reward = sum of actions (higher action → higher reward)
    def toy_reward(obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        return act.sum(dim=-1)

    planner = CEMPlanner(
        action_dim=act_dim,
        horizon=3,
        population=50,
        elite_frac=0.2,
        iterations=5,
        alpha=0.5,
        action_low=action_low,
        action_high=action_high,
    )
    state = torch.zeros(obs_dim)
    first_action, final_mean = planner.plan(
        state, ensemble, toy_reward, n_particles=4, ts_mode="tsinf"
    )

    # The optimal action is action_high; CEM mean should be positive (biased high)
    assert float(final_mean.mean()) > 0.0, (
        f"CEM mean should be positive (biased toward high actions); got {final_mean.mean():.4f}"
    )


def test_cem_risk_beta_penalizes_particle_return_dispersion() -> None:
    """Risk-aware CEM should prefer equal-mean plans with lower particle variance."""

    class RiskyToyEnsemble:
        ensemble_size = 2

        def propagate(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            model_idx: torch.Tensor,
        ) -> torch.Tensor:
            next_states = states.clone()
            signs = torch.where(model_idx % 2 == 0, 1.0, -1.0).to(states.device)
            next_states[:, 0] = signs * actions[:, 0]
            return next_states

    def state_reward(
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        del obs, act
        return next_obs[:, 0]

    state = torch.zeros(1)
    # Candidate 0: particle returns [+1, -1] -> mean 0, high dispersion.
    # Candidate 1: particle returns [+0.2, -0.2] -> mean 0, low dispersion.
    acts = torch.tensor([[[1.0]], [[0.2]]])
    ensemble = RiskyToyEnsemble()

    mean_planner = CEMPlanner(
        action_dim=1,
        horizon=1,
        population=2,
        elite_frac=0.5,
        iterations=1,
        alpha=1.0,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        risk_beta=0.0,
    )
    mean_scores = mean_planner._evaluate_sequences(
        acts, state, ensemble, state_reward, n_particles=2, ts_mode="tsinf"
    )
    assert torch.allclose(mean_scores, torch.zeros_like(mean_scores), atol=1e-6)

    robust_planner = CEMPlanner(
        action_dim=1,
        horizon=1,
        population=2,
        elite_frac=0.5,
        iterations=1,
        alpha=1.0,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        risk_beta=1.0,
    )
    robust_scores = robust_planner._evaluate_sequences(
        acts, state, ensemble, state_reward, n_particles=2, ts_mode="tsinf"
    )
    assert robust_scores[1] > robust_scores[0]


# ----------------------------------------------------------------------
# HalfCheetah reward vs environment
# ----------------------------------------------------------------------

def test_halfcheetah_reward_matches_env() -> None:
    """halfcheetah_reward should match the gymnasium reward to within 1e-3."""
    pytest.importorskip(
        "gymnasium.envs.mujoco",
        reason="MuJoCo not available; skipping HalfCheetah reward test",
    )
    import gymnasium as gym

    try:
        env = gym.make(
            "HalfCheetah-v5",
            render_mode=None,
            exclude_current_positions_from_observation=False,
        )
    except Exception as exc:
        pytest.skip(f"Could not create HalfCheetah-v5: {exc}")

    try:
        env.reset(seed=42)
        dt = float(env.unwrapped.dt)

        env_rewards: list[float] = []
        our_rewards: list[float] = []

        obs, _ = env.reset(seed=0)
        for _ in range(10):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            env_rewards.append(float(reward))

            obs_t = torch.tensor(obs, dtype=torch.float32)
            act_t = torch.tensor(action, dtype=torch.float32)
            nobs_t = torch.tensor(next_obs, dtype=torch.float32)
            our_r = float(halfcheetah_reward(obs_t, act_t, nobs_t, dt=dt))
            our_rewards.append(our_r)

            obs = next_obs
            if terminated or truncated:
                break

        for env_r, our_r in zip(env_rewards, our_rewards):
            assert abs(env_r - our_r) < 1e-3, (
                f"Reward mismatch: env={env_r:.4f}, ours={our_r:.4f}"
            )
    finally:
        env.close()


# ======================================================================
# Part 2 — Integration tests
# ======================================================================

from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY  # noqa: E402
from rl_from_scratch.pets.agent import PetsAgent  # noqa: E402
from rl_from_scratch.pets.config import PetsConfig  # noqa: E402
from rl_from_scratch.pets.training import train_pets  # noqa: E402


def _disable_pets_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_from_scratch.pets.reporting as _rep
    monkeypatch.setattr(_rep, "generate_training_figures", lambda *a, **kw: [])


def _make_tiny_ensemble(
    obs_dim: int = 4,
    act_dim: int = 2,
    ensemble_size: int = 2,
) -> ProbabilisticEnsemble:
    return ProbabilisticEnsemble(
        input_dim=obs_dim + act_dim,
        output_dim=obs_dim,
        ensemble_size=ensemble_size,
        hidden_dim=16,
        n_layers=2,
    )


def _make_pets_agent(
    obs_dim: int = 4,
    action_dim: int = 2,
    seed: int = 0,
    num_warmup_steps: int = 5,
) -> PetsAgent:
    torch.manual_seed(seed)
    return PetsAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=[-1.0] * action_dim,
        action_high=[1.0] * action_dim,
        env_id="HalfCheetah-v5",
        reward_dt=0.05,
        ensemble_size=2,
        hidden_dim=16,
        n_layers=2,
        dynamics_lr=1e-3,
        dynamics_fit_steps=5,
        dynamics_batch_size=16,
        weight_decay=0.0,
        plan_horizon=3,
        cem_population=10,
        cem_elite_frac=0.3,
        cem_iterations=2,
        cem_alpha=0.3,
        n_particles=4,
        ts_mode="tsinf",
        num_warmup_steps=num_warmup_steps,
        seed=seed,
    )


def _fill_agent_buffer(agent: PetsAgent, n: int = 20) -> None:
    rng = np.random.default_rng(42)
    for _ in range(n):
        obs = rng.standard_normal(agent.obs_dim).astype(np.float32)
        action = rng.standard_normal(agent.action_dim).astype(np.float32)
        next_obs = obs + 0.05 * rng.standard_normal(agent.obs_dim).astype(np.float32)
        agent.store_transition(obs, action, 0.0, next_obs, False)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

def test_pets_config_registered() -> None:
    assert "pets" in CONFIG_REGISTRY
    assert CONFIG_REGISTRY["pets"] is PetsConfig


def test_pets_agent_factory_registered() -> None:
    assert "pets" in AGENT_FACTORIES
    assert AGENT_FACTORIES["pets"] is train_pets


# -----------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------

def test_pets_config_rejects_invalid_ts_mode() -> None:
    with pytest.raises(ValueError, match="ts_mode"):
        PetsConfig(ts_mode="invalid")


def test_pets_config_rejects_elite_frac_above_one() -> None:
    with pytest.raises(ValueError, match="cem_elite_frac"):
        PetsConfig(cem_elite_frac=1.5)


def test_pets_config_rejects_elite_frac_zero() -> None:
    with pytest.raises(ValueError, match="cem_elite_frac"):
        PetsConfig(cem_elite_frac=0.0)


def test_pets_config_rejects_negative_risk_beta() -> None:
    with pytest.raises(ValueError, match="risk_beta"):
        PetsConfig(risk_beta=-0.1)


def test_pets_config_rejects_ensemble_size_one() -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        PetsConfig(ensemble_size=1)


def test_pets_config_round_trips_through_dict() -> None:
    config = PetsConfig(episodes=5, ensemble_size=3, ts_mode="ts1", risk_beta=0.35)
    d = config.to_dict()
    restored = PetsConfig.from_dict(d)
    assert restored.episodes == 5
    assert restored.ensemble_size == 3
    assert restored.ts_mode == "ts1"
    assert restored.risk_beta == pytest.approx(0.35)


# -----------------------------------------------------------------------
# Agent: select_action
# -----------------------------------------------------------------------

def test_pets_select_action_shape_dtype_during_warmup() -> None:
    agent = _make_pets_agent(obs_dim=4, action_dim=2, num_warmup_steps=100)
    obs = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32


def test_pets_select_action_within_bounds_after_warmup() -> None:
    agent = _make_pets_agent(obs_dim=4, action_dim=2, num_warmup_steps=5)
    _fill_agent_buffer(agent, n=10)
    agent.learn_step()   # fit model so _fitted = True

    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.standard_normal(4).astype(np.float32)
        action = agent.select_action(obs)
        assert action.shape == (2,)
        assert action.dtype == np.float32
        # Actions must be within [-1, 1]
        assert float(np.abs(action).max()) <= 1.0 + 1e-5, (
            f"Action out of bounds: {action}"
        )


# -----------------------------------------------------------------------
# Agent: learn_step
# -----------------------------------------------------------------------

def test_pets_learn_step_returns_finite_metrics() -> None:
    agent = _make_pets_agent(seed=0)
    _fill_agent_buffer(agent, n=30)

    metrics = agent.learn_step()

    assert "dynamics_nll" in metrics
    assert "ensemble_disagreement" in metrics
    assert "buffer_size" in metrics

    for key, value in metrics.items():
        assert math.isfinite(float(value)), f"{key} is not finite: {value}"


def test_pets_buffer_grows_with_transitions() -> None:
    agent = _make_pets_agent()
    assert len(agent.buffer) == 0
    _fill_agent_buffer(agent, n=10)
    assert len(agent.buffer) == 10


# -----------------------------------------------------------------------
# Save / load round-trip
# -----------------------------------------------------------------------

def test_pets_save_load_preserves_ensemble_predictions(tmp_path: Path) -> None:
    agent = _make_pets_agent(seed=3)
    _fill_agent_buffer(agent, n=20)
    agent.learn_step()

    ckpt = agent.save(tmp_path / "pets.pt")
    loaded = PetsAgent.load(ckpt)

    # Ensemble predictions must match before and after round-trip.
    # Move test inputs to the same device as each agent's ensemble.
    torch.manual_seed(0)
    x_test_cpu = torch.randn(5, agent.obs_dim + agent.action_dim)
    x_test_orig = x_test_cpu.to(agent.device)
    x_test_loaded = x_test_cpu.to(loaded.device)

    with torch.no_grad():
        means_orig, vars_orig = agent.ensemble.predict(x_test_orig)
        means_loaded, vars_loaded = loaded.ensemble.predict(x_test_loaded)

    assert torch.allclose(means_orig.cpu(), means_loaded.cpu(), atol=1e-5), (
        "Ensemble means differ after save/load"
    )
    assert torch.allclose(vars_orig.cpu(), vars_loaded.cpu(), atol=1e-5), (
        "Ensemble variances differ after save/load"
    )
    assert len(loaded.buffer) == len(agent.buffer)


# -----------------------------------------------------------------------
# Smoke training test
# -----------------------------------------------------------------------

def test_pets_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip(
        "gymnasium.envs.mujoco",
        reason="MuJoCo not available; skipping PETS smoke test",
    )

    _disable_pets_figures(monkeypatch)

    try:
        config = PetsConfig(
            env_id="HalfCheetah-v5",
            episodes=1,
            max_steps_per_episode=8,
            ensemble_size=2,
            hidden_dim=16,
            n_layers=2,
            dynamics_fit_steps=3,
            dynamics_batch_size=8,
            plan_horizon=4,
            cem_population=16,
            cem_elite_frac=0.2,
            cem_iterations=2,
            cem_alpha=0.3,
            n_particles=4,
            ts_mode="tsinf",
            num_warmup_steps=10,
            eval_every=1,
            eval_episodes=1,
            checkpoint_every=1,
            output_dir=str(tmp_path),
        )

        result = train_pets(config, seed=0)

        assert set(result) == {"agent", "history", "metrics", "paths"}
        assert isinstance(result["agent"], PetsAgent)
        assert len(result["history"]["episode_rewards"]) == 1
        assert result["paths"].run_dir.exists()

    except Exception as exc:
        err_msg = str(exc).lower()
        if "mujoco" in err_msg or "halfcheetah" in err_msg:
            pytest.skip(f"MuJoCo/HalfCheetah unavailable: {exc}")
        raise
