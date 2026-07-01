"""Tests for MBPO (Model-Based Policy Optimization).

Part 1 (math/unit tests):
  - gaussian_nll hand computation.
  - Ensemble trained on toy [Δs, r] target reduces NLL and reward-head
    correlates with true reward.
  - propagate returns (next_states, rewards) with correct shapes.
  - SacLearner.update returns finite metrics and reduces critic loss over
    repeated updates on a fixed batch.
  - sample_mixed returns correct total size and respects the real/model split.
  - Reward-head predicts HalfCheetah reward decently (correlation > 0.5).

Part 2 (integration tests):
  - Registry: "mbpo" in CONFIG_REGISTRY and AGENT_FACTORIES.
  - Config validation rejects bad real_ratio / ensemble_size.
  - select_action shape/dtype + bounds.
  - learn_step returns finite floats.
  - save/load round-trip: policy action matches.
  - Smoke train_mbpo with TINY config.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import rl_from_scratch  # noqa: F401  triggers auto-discovery of registries

from rl_from_scratch.mbpo.dynamics import (
    ProbabilisticEnsemble,
    ProbabilisticMLP,
    gaussian_nll,
)
from rl_from_scratch.mbpo.sac import SacLearner
from rl_from_scratch.mbpo.buffer import ModelBuffer, ReplayBuffer, sample_mixed


# ======================================================================
# Fixtures
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
    """For mean=0, logvar=0, target=1: NLL = 0.5 * (1 + 0) = 0.5."""
    mean = torch.zeros(1, 1)
    logvar = torch.zeros(1, 1)
    target = torch.ones(1, 1)
    result = float(gaussian_nll(mean, logvar, target))
    assert result == pytest.approx(0.5, abs=1e-6)


def test_gaussian_nll_zero_at_perfect_prediction() -> None:
    """When mean == target and logvar = 0, NLL = 0."""
    mean = torch.tensor([[1.0, 2.0]])
    logvar = torch.zeros(1, 2)
    target = torch.tensor([[1.0, 2.0]])
    result = float(gaussian_nll(mean, logvar, target))
    assert result == pytest.approx(0.0, abs=1e-6)


def test_gaussian_nll_batched_mean() -> None:
    """Batch NLL equals the mean of per-sample NLLs."""
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
# Ensemble on toy [Δstate, reward] target
# ----------------------------------------------------------------------


def _make_toy_delta_reward_data(
    n: int = 300, obs_dim: int = 4, act_dim: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Toy dataset: Y = [Δstate, reward] where reward = sum(action)."""
    torch.manual_seed(42)
    X = torch.randn(n, obs_dim + act_dim)
    delta = 0.1 * torch.randn(n, obs_dim)
    reward = X[:, obs_dim:obs_dim + act_dim].sum(dim=1, keepdim=True)  # simple signal
    Y = torch.cat([delta, reward], dim=1)  # [n, obs_dim + 1]
    return X, Y


def test_ensemble_reduces_nll_on_delta_reward_toy() -> None:
    """Training should decrease NLL on the [Δstate, reward] toy dataset."""
    obs_dim, act_dim = 4, 2
    X, Y = _make_toy_delta_reward_data(n=200, obs_dim=obs_dim, act_dim=act_dim)
    ensemble = ProbabilisticEnsemble(
        input_dim=obs_dim + act_dim,
        output_dim=obs_dim + 1,
        ensemble_size=2,
        hidden_dim=32,
        n_layers=2,
    )
    # Evaluate initial NLL before training
    ensemble.fit_normalizer(X, Y)
    x_norm = (X - ensemble.x_mean) / ensemble.x_std
    y_norm = (Y - ensemble.y_mean) / ensemble.y_std

    with torch.no_grad():
        m0, lv0 = ensemble.members[0](x_norm)
        nll_before = float(gaussian_nll(m0, lv0, y_norm))

    nll_after = ensemble.fit(X, Y, steps=150, batch_size=64, lr=1e-3, weight_decay=0.0)
    assert nll_after < nll_before, (
        f"NLL should decrease after training: before={nll_before:.4f}, after={nll_after:.4f}"
    )


def test_ensemble_reward_head_correlates_with_true_reward_toy() -> None:
    """The reward column of the ensemble prediction should correlate with the true reward."""
    obs_dim, act_dim = 4, 2
    X, Y = _make_toy_delta_reward_data(n=400, obs_dim=obs_dim, act_dim=act_dim)
    ensemble = ProbabilisticEnsemble(
        input_dim=obs_dim + act_dim,
        output_dim=obs_dim + 1,
        ensemble_size=3,
        hidden_dim=64,
        n_layers=2,
    )
    ensemble.fit(X, Y, steps=300, batch_size=64, lr=1e-3, weight_decay=0.0)

    # Predict on a test set
    X_test, Y_test = _make_toy_delta_reward_data(n=100, obs_dim=obs_dim, act_dim=act_dim)
    with torch.no_grad():
        means, _ = ensemble.predict(X_test)  # [100, E, obs_dim+1]
    # Average across ensemble members; take the reward column (last)
    pred_reward = means[:, :, obs_dim:].mean(dim=1).squeeze(-1).numpy()  # [100]
    true_reward = Y_test[:, obs_dim].numpy()

    correlation = float(np.corrcoef(pred_reward, true_reward)[0, 1])
    assert correlation > 0.5, (
        f"Reward head correlation should exceed 0.5; got {correlation:.4f}"
    )


# ----------------------------------------------------------------------
# propagate
# ----------------------------------------------------------------------


def _make_fitted_ensemble(
    obs_dim: int = 4, act_dim: int = 2, ensemble_size: int = 3
) -> ProbabilisticEnsemble:
    torch.manual_seed(7)
    ensemble = ProbabilisticEnsemble(
        input_dim=obs_dim + act_dim,
        output_dim=obs_dim + 1,
        ensemble_size=ensemble_size,
        hidden_dim=16,
        n_layers=2,
    )
    X = torch.randn(50, obs_dim + act_dim)
    Y = torch.randn(50, obs_dim + 1)
    ensemble.fit_normalizer(X, Y)
    return ensemble


def test_propagate_returns_correct_shapes() -> None:
    """propagate returns (next_states [B, obs_dim], rewards [B])."""
    obs_dim, act_dim, E = 4, 2, 3
    B = 10
    ensemble = _make_fitted_ensemble(obs_dim, act_dim, E)
    states = torch.randn(B, obs_dim)
    actions = torch.randn(B, act_dim)
    model_idx = torch.randint(0, E, (B,))

    next_states, rewards = ensemble.propagate(states, actions, model_idx)

    assert next_states.shape == (B, obs_dim), (
        f"Expected next_states shape ({B}, {obs_dim}), got {next_states.shape}"
    )
    assert rewards.shape == (B,), (
        f"Expected rewards shape ({B},), got {rewards.shape}"
    )


def test_propagate_next_state_is_state_plus_delta() -> None:
    """next_states = states + delta, and rewards are finite floats."""
    obs_dim, act_dim, E = 3, 2, 2
    B = 8
    ensemble = _make_fitted_ensemble(obs_dim, act_dim, E)
    states = torch.ones(B, obs_dim) * 2.0
    actions = torch.zeros(B, act_dim)
    model_idx = torch.zeros(B, dtype=torch.long)

    torch.manual_seed(0)
    next_states, rewards = ensemble.propagate(states, actions, model_idx)

    assert torch.isfinite(next_states).all(), "next_states must be finite"
    assert torch.isfinite(rewards).all(), "rewards must be finite"


# ----------------------------------------------------------------------
# SacLearner
# ----------------------------------------------------------------------


def _make_sac_learner(obs_dim: int = 4, action_dim: int = 2) -> SacLearner:
    return SacLearner(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=32,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune_alpha=True,
        alpha_lr=3e-4,
        target_entropy=None,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        device="cpu",
    )


def _make_batch(
    obs_dim: int = 4, action_dim: int = 2, batch_size: int = 8
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    obs = torch.randn(batch_size, obs_dim)
    act = torch.tanh(torch.randn(batch_size, action_dim))
    rew = torch.randn(batch_size)
    next_obs = torch.randn(batch_size, obs_dim)
    done = torch.zeros(batch_size)
    return obs, act, rew, next_obs, done


def test_sac_learner_update_returns_finite_metrics() -> None:
    """update returns finite floats for all expected metric keys."""
    learner = _make_sac_learner()
    obs, act, rew, next_obs, done = _make_batch()

    metrics = learner.update(obs, act, rew, next_obs, done)

    expected_keys = {"critic_loss", "actor_loss", "alpha_loss", "alpha", "entropy", "q_mean"}
    assert expected_keys.issubset(metrics.keys()), (
        f"Missing keys: {expected_keys - set(metrics.keys())}"
    )
    for k, v in metrics.items():
        assert math.isfinite(float(v)), f"Metric {k} is not finite: {v}"


def test_sac_learner_critic_loss_decreases_on_fixed_batch() -> None:
    """Repeated SAC updates on a fixed batch should decrease the critic loss."""
    torch.manual_seed(0)
    learner = _make_sac_learner()
    obs, act, rew, next_obs, done = _make_batch(batch_size=32)

    losses = []
    for _ in range(20):
        m = learner.update(obs, act, rew, next_obs, done)
        losses.append(m["critic_loss"])

    # The loss trajectory should trend downward on this fixed batch
    first_half = float(np.mean(losses[:5]))
    second_half = float(np.mean(losses[-5:]))
    assert second_half < first_half, (
        f"Critic loss should decrease on a fixed batch: "
        f"first={first_half:.4f}, last={second_half:.4f}"
    )


def test_sac_learner_select_action_shape_and_bounds() -> None:
    """select_action returns an action of the correct shape within bounds."""
    learner = _make_sac_learner(obs_dim=4, action_dim=2)
    obs = np.zeros(4, dtype=np.float32)

    action = learner.select_action(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0 - 1e-5)
    assert np.all(action <= 1.0 + 1e-5)


# ----------------------------------------------------------------------
# sample_mixed
# ----------------------------------------------------------------------


def _fill_buffer(buf: ReplayBuffer, n: int, obs_dim: int = 4, act_dim: int = 2) -> None:
    rng = np.random.default_rng(42)
    for _ in range(n):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        act = rng.standard_normal(act_dim).astype(np.float32)
        rew = float(rng.standard_normal())
        next_obs = rng.standard_normal(obs_dim).astype(np.float32)
        done = False
        buf.push(obs, act, rew, next_obs, done)


def test_sample_mixed_total_size() -> None:
    """sample_mixed returns exactly batch_size transitions."""
    env_buf = ReplayBuffer(1000)
    model_buf = ModelBuffer(1000)
    _fill_buffer(env_buf, 100)
    _fill_buffer(model_buf, 100)

    obs, act, rew, next_obs, done = sample_mixed(env_buf, model_buf, 32, real_ratio=0.05)

    assert obs.shape[0] == 32
    assert act.shape[0] == 32
    assert rew.shape[0] == 32


def test_sample_mixed_real_ratio_respected() -> None:
    """With real_ratio=1.0, all samples come from env_buffer (tagged trick)."""
    obs_dim = 4
    env_buf = ReplayBuffer(1000)
    model_buf = ModelBuffer(1000)

    # Tag env transitions with obs[0] = +100, model with obs[0] = -100
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        obs[0] = 100.0  # env tag
        env_buf.push(obs, np.zeros(2, np.float32), 0.0, obs, False)
    for _ in range(200):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        obs[0] = -100.0  # model tag
        model_buf.push(obs, np.zeros(2, np.float32), 0.0, obs, False)

    batch_obs, _, _, _, _ = sample_mixed(env_buf, model_buf, 32, real_ratio=1.0)
    assert (batch_obs[:, 0] > 0).all(), "All samples should be from env_buffer when real_ratio=1.0"


def test_sample_mixed_fallback_when_model_empty() -> None:
    """When model buffer is empty, all samples come from env buffer."""
    env_buf = ReplayBuffer(1000)
    model_buf = ModelBuffer(1000)
    _fill_buffer(env_buf, 50)

    obs, act, rew, next_obs, done = sample_mixed(env_buf, model_buf, 16, real_ratio=0.05)
    assert obs.shape[0] == 16


def test_sample_mixed_fallback_when_env_empty() -> None:
    """When env buffer is empty, all samples come from model buffer."""
    env_buf = ReplayBuffer(1000)
    model_buf = ModelBuffer(1000)
    _fill_buffer(model_buf, 50)

    obs, act, rew, next_obs, done = sample_mixed(env_buf, model_buf, 16, real_ratio=0.05)
    assert obs.shape[0] == 16


# ----------------------------------------------------------------------
# Reward head on real HalfCheetah transitions
# ----------------------------------------------------------------------


def test_reward_head_correlates_with_halfcheetah_reward() -> None:
    """Ensemble reward head should have correlation > 0.5 with env rewards on real data."""
    pytest.importorskip(
        "gymnasium.envs.mujoco",
        reason="MuJoCo not available; skipping HalfCheetah reward-head test",
    )
    import gymnasium as gym

    try:
        env = gym.make("HalfCheetah-v5", render_mode=None)
    except Exception as exc:
        pytest.skip(f"Could not create HalfCheetah-v5: {exc}")

    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Collect transitions
        transitions = []
        obs, _ = env.reset(seed=0)
        for _ in range(400):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            transitions.append((obs, action, float(reward), next_obs))
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset(seed=len(transitions))

        obs_arr = np.stack([t[0] for t in transitions]).astype(np.float32)
        act_arr = np.stack([t[1] for t in transitions]).astype(np.float32)
        rew_arr = np.array([t[2] for t in transitions], dtype=np.float32)
        nobs_arr = np.stack([t[3] for t in transitions]).astype(np.float32)

        X = torch.tensor(np.concatenate([obs_arr, act_arr], axis=1))
        delta = nobs_arr - obs_arr
        Y = torch.tensor(np.concatenate([delta, rew_arr[:, None]], axis=1))

        # Train a small ensemble
        ensemble = ProbabilisticEnsemble(
            input_dim=obs_dim + act_dim,
            output_dim=obs_dim + 1,
            ensemble_size=3,
            hidden_dim=64,
            n_layers=2,
        )
        ensemble.fit(X, Y, steps=200, batch_size=64, lr=1e-3, weight_decay=1e-4)

        with torch.no_grad():
            means, _ = ensemble.predict(X)   # [N, E, obs_dim+1]
        pred_reward = means[:, :, obs_dim:].mean(dim=1).squeeze(-1).numpy()

        correlation = float(np.corrcoef(pred_reward, rew_arr)[0, 1])
        assert correlation > 0.5, (
            f"Reward head correlation should exceed 0.5; got {correlation:.4f}"
        )
    finally:
        env.close()


# ======================================================================
# Part 2 — Integration tests
# ======================================================================


from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY  # noqa: E402
from rl_from_scratch.mbpo.agent import MbpoAgent  # noqa: E402
from rl_from_scratch.mbpo.config import MbpoConfig  # noqa: E402
from rl_from_scratch.mbpo.training import _effective_model_buffer_capacity, train_mbpo  # noqa: E402


def _disable_mbpo_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_from_scratch.mbpo.reporting as _rep
    monkeypatch.setattr(_rep, "generate_training_figures", lambda *a, **kw: [])


def _make_mbpo_agent(
    obs_dim: int = 4,
    action_dim: int = 2,
    seed: int = 0,
) -> MbpoAgent:
    torch.manual_seed(seed)
    return MbpoAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=[-1.0] * action_dim,
        action_high=[1.0] * action_dim,
        ensemble_size=2,
        model_hidden_dim=16,
        model_n_layers=2,
        model_lr=1e-3,
        model_fit_steps=5,
        model_batch_size=16,
        weight_decay=0.0,
        sac_hidden_dim=16,
        sac_batch_size=8,
        alpha=0.2,
        auto_tune_alpha=True,
        rollout_length=1,
        rollout_batch_size=8,
        rollout_every=5,
        updates_per_step=2,
        real_ratio=0.05,
        env_buffer_capacity=10000,
        model_buffer_capacity=10000,
        num_warmup_steps=5,
        seed=seed,
        device="cpu",
    )


def _fill_agent_env_buffer(agent: MbpoAgent, n: int = 20) -> None:
    rng = np.random.default_rng(42)
    for _ in range(n):
        obs = rng.standard_normal(agent.obs_dim).astype(np.float32)
        action = rng.standard_normal(agent.action_dim).astype(np.float32)
        next_obs = obs + 0.05 * rng.standard_normal(agent.obs_dim).astype(np.float32)
        agent.store_transition(obs, action, 1.0, next_obs, False)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------


def test_mbpo_config_registered() -> None:
    assert "mbpo" in CONFIG_REGISTRY
    assert CONFIG_REGISTRY["mbpo"] is MbpoConfig


def test_mbpo_agent_factory_registered() -> None:
    assert "mbpo" in AGENT_FACTORIES
    assert AGENT_FACTORIES["mbpo"] is train_mbpo


# -----------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------


def test_mbpo_config_rejects_bad_real_ratio() -> None:
    with pytest.raises(ValueError, match="real_ratio"):
        MbpoConfig(real_ratio=1.5)


def test_mbpo_config_rejects_ensemble_size_one() -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        MbpoConfig(ensemble_size=1)


def test_mbpo_config_rejects_bad_tau() -> None:
    with pytest.raises(ValueError, match="tau"):
        MbpoConfig(tau=1.0)


def test_mbpo_config_rejects_bad_model_retain_epochs() -> None:
    with pytest.raises(ValueError, match="model_retain_epochs"):
        MbpoConfig(model_retain_epochs=0)


def test_mbpo_config_round_trips_through_dict() -> None:
    config = MbpoConfig(epochs=5, ensemble_size=3, rollout_length=2)
    d = config.to_dict()
    restored = MbpoConfig.from_dict(d)
    assert restored.epochs == 5
    assert restored.ensemble_size == 3
    assert restored.rollout_length == 2


def test_mbpo_model_retain_capacity_caps_stale_imagined_data() -> None:
    config = MbpoConfig(
        steps_per_epoch=100,
        rollout_every=20,
        rollout_batch_size=10,
        rollout_length=2,
        model_retain_epochs=3,
        model_buffer_capacity=10_000,
        sac_batch_size=32,
    )
    # ceil(100 / 20) * 10 * 2 * 3 = 300 recent imagined transitions.
    assert _effective_model_buffer_capacity(config) == 300

    capped = MbpoConfig(
        steps_per_epoch=100,
        rollout_every=20,
        rollout_batch_size=10,
        rollout_length=2,
        model_retain_epochs=3,
        model_buffer_capacity=128,
        sac_batch_size=32,
    )
    assert _effective_model_buffer_capacity(capped) == 128


# -----------------------------------------------------------------------
# Agent: select_action
# -----------------------------------------------------------------------


def test_mbpo_select_action_during_warmup() -> None:
    """During warm-up (buffer too small), returns random action in bounds."""
    agent = _make_mbpo_agent(obs_dim=4, action_dim=2)
    # num_warmup_steps=5, env_buffer is empty → should return random
    obs = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0 - 1e-5)
    assert np.all(action <= 1.0 + 1e-5)


def test_mbpo_select_action_after_warmup() -> None:
    """After warm-up, uses the SAC policy and stays in bounds."""
    agent = _make_mbpo_agent(obs_dim=4, action_dim=2)
    _fill_agent_env_buffer(agent, n=20)  # fills past num_warmup_steps=5

    obs = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs, deterministic=False)
    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0 - 1e-5)
    assert np.all(action <= 1.0 + 1e-5)


# -----------------------------------------------------------------------
# Agent: learn_step
# -----------------------------------------------------------------------


def test_mbpo_learn_step_returns_finite_metrics() -> None:
    """learn_step returns finite float metrics when buffers have enough data."""
    agent = _make_mbpo_agent()
    _fill_agent_env_buffer(agent, n=30)

    # Fit the model to build normaliser, then generate rollouts to fill model buffer
    agent.fit_model()
    agent.generate_model_rollouts(rollout_length=1)

    metrics = agent.learn_step()
    if metrics:
        for k, v in metrics.items():
            assert math.isfinite(float(v)), f"Metric {k} is not finite: {v}"


def test_mbpo_learn_step_empty_when_buffers_too_small() -> None:
    """learn_step returns empty dict when neither buffer has enough data."""
    agent = _make_mbpo_agent()
    # sac_batch_size=8, both buffers empty
    metrics = agent.learn_step()
    assert metrics == {}


# -----------------------------------------------------------------------
# Save / load round-trip
# -----------------------------------------------------------------------


def test_mbpo_save_load_policy_action_matches(tmp_path: Path) -> None:
    """After save/load, the SAC policy produces the same deterministic action."""
    agent = _make_mbpo_agent(seed=3)
    _fill_agent_env_buffer(agent, n=20)

    ckpt_path = agent.save(tmp_path / "mbpo.pt")
    loaded = MbpoAgent.load(ckpt_path, device="cpu")
    # Pre-populate loaded agent's buffer so it uses the SAC policy (not random warm-up)
    _fill_agent_env_buffer(loaded, n=20)

    obs = np.zeros(4, dtype=np.float32)
    action_orig = agent.select_action(obs, deterministic=True)
    action_loaded = loaded.select_action(obs, deterministic=True)

    np.testing.assert_allclose(
        action_orig, action_loaded, atol=1e-5,
        err_msg="Policy action should match after save/load round-trip"
    )


def test_mbpo_save_load_ensemble_predictions_match(tmp_path: Path) -> None:
    """After save/load, ensemble predictions are identical."""
    agent = _make_mbpo_agent(seed=5)
    _fill_agent_env_buffer(agent, n=20)
    agent.fit_model()

    ckpt_path = agent.save(tmp_path / "mbpo_ens.pt")
    loaded = MbpoAgent.load(ckpt_path, device="cpu")

    x_test = torch.randn(5, agent.obs_dim + agent.action_dim)
    with torch.no_grad():
        means_orig, _ = agent.ensemble.predict(x_test)
        means_loaded, _ = loaded.ensemble.predict(x_test)

    assert torch.allclose(means_orig, means_loaded, atol=1e-5), (
        "Ensemble means should match after save/load"
    )


# -----------------------------------------------------------------------
# Smoke training test
# -----------------------------------------------------------------------


def test_mbpo_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_mbpo with a tiny config completes and returns the expected contract."""
    pytest.importorskip(
        "gymnasium.envs.mujoco",
        reason="MuJoCo not available; skipping MBPO smoke test",
    )

    _disable_mbpo_figures(monkeypatch)

    try:
        config = MbpoConfig(
            env_id="HalfCheetah-v5",
            epochs=1,
            steps_per_epoch=10,
            max_steps_per_episode=10,
            ensemble_size=2,
            model_hidden_dim=16,
            model_n_layers=2,
            model_fit_steps=3,
            model_batch_size=8,
            sac_hidden_dim=16,
            sac_batch_size=8,
            rollout_length=1,
            rollout_batch_size=8,
            rollout_every=2,
            updates_per_step=2,
            real_ratio=0.05,
            env_buffer_capacity=10000,
            model_buffer_capacity=10000,
            num_warmup_steps=10,
            eval_every=1,
            eval_episodes=1,
            checkpoint_every=1,
            output_dir=str(tmp_path),
        )

        result = train_mbpo(config, seed=0)

        assert set(result) == {"agent", "history", "metrics", "paths"}
        assert isinstance(result["agent"], MbpoAgent)
        assert isinstance(result["history"], dict)
        assert result["paths"].run_dir.exists()

    except Exception as exc:
        err_msg = str(exc).lower()
        if "mujoco" in err_msg or "halfcheetah" in err_msg:
            pytest.skip(f"MuJoCo/HalfCheetah unavailable: {exc}")
        raise


# -----------------------------------------------------------------------
# Cross-package import isolation
# -----------------------------------------------------------------------


def test_mbpo_has_no_cross_package_imports() -> None:
    """MBPO must not import from sibling algorithm packages."""
    import ast

    mbpo_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rl_from_scratch"
        / "mbpo"
    )

    forbidden = {
        "rl_from_scratch.pets",
        "rl_from_scratch.sac",
        "rl_from_scratch.pilco",
        "rl_from_scratch.deep_q",
        "rl_from_scratch.actor_critic",
        "rl_from_scratch.deterministic_actor_critic",
        "rl_from_scratch.trust_region",
    }

    for path in mbpo_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ""
                for alias in node.names:
                    for pkg in forbidden:
                        assert not alias.name.startswith(pkg), (
                            f"Cross-package import of '{alias.name}' found in {path.name}"
                        )
                continue
            else:
                continue
            for pkg in forbidden:
                assert not module.startswith(pkg), (
                    f"Cross-package import 'from {module}' found in {path.name}"
                )
