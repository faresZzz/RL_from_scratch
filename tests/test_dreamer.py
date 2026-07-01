"""Tests for DreamerV1 (Dream to Control, Hafner et al. 2020).

Part 1 (math/unit tests):
  - RSSM obs_step / img_step output shapes.
  - RSSM get_feat shape.
  - kl_loss >= 0 and respects free_nats.
  - World-model reduces recon loss on a tiny toy sequence.
  - Reward head correlates with targets after fitting.
  - Actor.sample returns actions within [low, high] (±1e-5) and finite log_prob.
  - Imagination rollout shapes.
  - lambda_returns finite & correct shape.
  - SequenceBuffer: add episodes, sample, raises on missing seq_len.

Part 2 (integration tests):
  - Registry: "dreamer" in CONFIG_REGISTRY and AGENT_FACTORIES.
  - Config validation: bad lambda_, imagination_horizon, dims.
  - to_dict / from_dict round-trip.
  - select_action shape (6,), dtype float32, within bounds.
  - learn_step returns all-finite floats.
  - save/load round-trip: deterministic action matches to atol=1e-5.
  - Smoke train_dreamer with tiny config.
  - Independence test (AST walk over dreamer/*.py).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import rl_from_scratch  # noqa: F401  triggers auto-discovery of registries

from rl_from_scratch.dreamer.rssm import RSSM
from rl_from_scratch.dreamer.networks import Encoder, Decoder, RewardModel
from rl_from_scratch.dreamer.actor_critic import Actor, Critic
from rl_from_scratch.dreamer.buffer import SequenceBuffer
from rl_from_scratch.dreamer.agent import lambda_returns


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

# ──────────────────────────────────────────────────────────────────────
# RSSM
# ──────────────────────────────────────────────────────────────────────


def _make_rssm(
    action_dim: int = 3,
    embed_dim: int = 16,
    deter_dim: int = 20,
    stoch_dim: int = 8,
    hidden_dim: int = 16,
) -> RSSM:
    torch.manual_seed(0)
    return RSSM(
        action_dim=action_dim,
        embed_dim=embed_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
        min_std=0.1,
    )


def test_rssm_obs_step_output_shapes() -> None:
    """obs_step returns (post, prior) each with correct shapes."""
    B, action_dim, embed_dim = 4, 3, 16
    deter_dim, stoch_dim = 20, 8
    rssm = _make_rssm(action_dim, embed_dim, deter_dim, stoch_dim)

    state = rssm.initial_state(B)
    prev_action = torch.randn(B, action_dim)
    embed = torch.randn(B, embed_dim)

    post, prior = rssm.obs_step(state, prev_action, embed)

    for name, s in [("post", post), ("prior", prior)]:
        assert s["deter"].shape == (B, deter_dim), f"{name}.deter shape wrong"
        assert s["stoch"].shape == (B, stoch_dim), f"{name}.stoch shape wrong"
        assert s["mean"].shape == (B, stoch_dim), f"{name}.mean shape wrong"
        assert s["std"].shape == (B, stoch_dim), f"{name}.std shape wrong"
        assert (s["std"] > 0).all(), f"{name}.std must be positive"


def test_rssm_img_step_output_shapes() -> None:
    """img_step returns a prior dict with correct shapes."""
    B, action_dim, embed_dim = 4, 3, 16
    deter_dim, stoch_dim = 20, 8
    rssm = _make_rssm(action_dim, embed_dim, deter_dim, stoch_dim)

    state = rssm.initial_state(B)
    prev_action = torch.randn(B, action_dim)
    prior = rssm.img_step(state, prev_action)

    assert prior["deter"].shape == (B, deter_dim)
    assert prior["stoch"].shape == (B, stoch_dim)
    assert (prior["std"] > 0).all()


def test_rssm_get_feat_shape() -> None:
    """get_feat returns cat([deter, stoch]) with shape [B, deter+stoch]."""
    B, deter_dim, stoch_dim = 4, 20, 8
    rssm = _make_rssm(deter_dim=deter_dim, stoch_dim=stoch_dim)
    state = rssm.initial_state(B)
    feat = rssm.get_feat(state)
    assert feat.shape == (B, deter_dim + stoch_dim)


def test_rssm_kl_loss_nonnegative() -> None:
    """KL divergence is always >= 0."""
    B, deter_dim, stoch_dim = 4, 20, 8
    rssm = _make_rssm(deter_dim=deter_dim, stoch_dim=stoch_dim)
    state = rssm.initial_state(B)
    embed = torch.randn(B, 16)
    action = torch.randn(B, 3)
    post, prior = rssm.obs_step(state, action, embed)
    kl = rssm.kl_loss(post, prior, free_nats=0.0)
    assert float(kl.detach()) >= 0.0


def test_rssm_kl_loss_respects_free_nats() -> None:
    """With a huge free_nats, kl_loss == free_nats."""
    B, deter_dim, stoch_dim = 4, 20, 8
    rssm = _make_rssm(deter_dim=deter_dim, stoch_dim=stoch_dim)
    state = rssm.initial_state(B)
    embed = torch.randn(B, 16)
    action = torch.randn(B, 3)
    post, prior = rssm.obs_step(state, action, embed)
    huge = 1e9
    kl = rssm.kl_loss(post, prior, free_nats=huge)
    assert float(kl.detach()) == pytest.approx(huge, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────
# World-model: recon loss decreases
# ──────────────────────────────────────────────────────────────────────


def test_world_model_recon_loss_decreases_on_toy_sequence() -> None:
    """Training the world-model should reduce reconstruction loss on a toy fixed seq."""
    torch.manual_seed(42)
    B, L, obs_dim, action_dim = 4, 6, 8, 3
    embed_dim, deter_dim, stoch_dim, hidden_dim = 16, 20, 8, 16

    obs = torch.randn(B, L, obs_dim)
    action = torch.randn(B, L, action_dim)

    encoder = Encoder(obs_dim, hidden_dim, embed_dim)
    rssm = RSSM(action_dim, embed_dim, deter_dim, stoch_dim, hidden_dim)
    decoder = Decoder(deter_dim + stoch_dim, hidden_dim, obs_dim)

    params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    def _recon_loss() -> float:
        embed = encoder(obs)
        prev_a = torch.cat([torch.zeros(B, 1, action_dim), action[:, :-1]], dim=1)
        state = rssm.initial_state(B)
        feats = []
        for t in range(L):
            post, _ = rssm.obs_step(state, prev_a[:, t], embed[:, t])
            feats.append(rssm.get_feat(post))
            state = post
        feats_t = torch.stack(feats, dim=1)
        obs_pred = decoder(feats_t)
        return float((0.5 * ((obs_pred - obs) ** 2).sum(-1).mean()).detach())

    loss_before = _recon_loss()

    for _ in range(50):
        optim.zero_grad()
        embed = encoder(obs)
        prev_a = torch.cat([torch.zeros(B, 1, action_dim), action[:, :-1]], dim=1)
        state = rssm.initial_state(B)
        feats = []
        for t in range(L):
            post, _ = rssm.obs_step(state, prev_a[:, t], embed[:, t])
            feats.append(rssm.get_feat(post))
            state = post
        feats_t = torch.stack(feats, dim=1)
        obs_pred = decoder(feats_t)
        loss = 0.5 * ((obs_pred - obs) ** 2).sum(-1).mean()
        loss.backward()
        optim.step()

    loss_after = _recon_loss()
    assert loss_after < loss_before, (
        f"Recon loss should decrease: before={loss_before:.4f}, after={loss_after:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────
# Reward head: correlation
# ──────────────────────────────────────────────────────────────────────


def test_reward_head_correlates_after_fitting() -> None:
    """After training the reward model on a toy linear signal, correlation > 0.7."""
    torch.manual_seed(7)
    N, feat_dim, hidden_dim = 300, 12, 32
    feats = torch.randn(N, feat_dim)
    # Simple linear target: reward = sum of first 3 dims
    true_reward = feats[:, :3].sum(dim=-1)

    model = RewardModel(feat_dim, hidden_dim)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(200):
        optim.zero_grad()
        pred = model(feats).squeeze(-1)
        loss = 0.5 * ((pred - true_reward) ** 2).mean()
        loss.backward()
        optim.step()

    with torch.no_grad():
        pred = model(feats).squeeze(-1).numpy()
    true = true_reward.numpy()
    corr = float(np.corrcoef(pred, true)[0, 1])
    assert corr > 0.7, f"Reward correlation should exceed 0.7, got {corr:.4f}"


# ──────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────


def test_actor_sample_within_bounds() -> None:
    """Actor.sample returns actions in [low, high] ± 1e-5."""
    torch.manual_seed(0)
    feat_dim, action_dim, hidden_dim = 28, 6, 32
    low = np.full(action_dim, -1.0, dtype=np.float32)
    high = np.full(action_dim, 1.0, dtype=np.float32)

    actor = Actor(feat_dim, action_dim, hidden_dim, action_low=low, action_high=high)
    feat = torch.randn(16, feat_dim)
    action, log_prob = actor.sample(feat)

    assert action.shape == (16, action_dim)
    assert torch.all(action >= torch.tensor(low) - 1e-5)
    assert torch.all(action <= torch.tensor(high) + 1e-5)
    assert torch.isfinite(log_prob).all(), "log_prob must be finite"


def test_actor_sample_log_prob_finite() -> None:
    """log_prob from Actor.sample is always finite."""
    torch.manual_seed(1)
    actor = Actor(16, 4, 32, action_low=-1.0, action_high=1.0)
    feat = torch.randn(64, 16)
    _, log_prob = actor.sample(feat)
    assert torch.isfinite(log_prob).all()


# ──────────────────────────────────────────────────────────────────────
# lambda_returns
# ──────────────────────────────────────────────────────────────────────


def test_lambda_returns_shape() -> None:
    """lambda_returns returns [T-1, M] tensor."""
    T, M = 6, 8
    rewards = torch.randn(T, M)
    values = torch.randn(T, M)
    rets = lambda_returns(rewards, values, gamma=0.99, lam=0.95)
    assert rets.shape == (T - 1, M)


def test_lambda_returns_finite() -> None:
    """lambda_returns produces only finite values."""
    T, M = 5, 4
    rewards = torch.randn(T, M)
    values = torch.randn(T, M)
    rets = lambda_returns(rewards, values, gamma=0.99, lam=0.95)
    assert torch.isfinite(rets).all()


def test_lambda_returns_lam0_is_one_step_td() -> None:
    """λ=0 → λ-return = r_{t+1} + γ * V_{t+1}."""
    T, M = 4, 3
    rewards = torch.ones(T, M)
    values = torch.ones(T, M) * 2.0
    gamma = 0.9
    rets = lambda_returns(rewards, values, gamma=gamma, lam=0.0)
    # For λ=0: G_t = r_{t+1} + γ * V_{t+1}
    expected = rewards[1:] + gamma * values[1:]  # [T-1, M]
    assert torch.allclose(rets, expected, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# Imagination rollout shapes
# ──────────────────────────────────────────────────────────────────────


def test_imagination_rollout_shapes() -> None:
    """Imagination rollout produces [H+1, M, F] feature tensor."""
    torch.manual_seed(0)
    action_dim, embed_dim = 3, 16
    deter_dim, stoch_dim, hidden_dim = 20, 8, 16
    M, H = 8, 5

    rssm = _make_rssm(action_dim, embed_dim, deter_dim, stoch_dim, hidden_dim)
    actor = Actor(deter_dim + stoch_dim, action_dim, hidden_dim)

    state = rssm.initial_state(M)
    feats = [rssm.get_feat(state)]

    for _ in range(H):
        act, _ = actor.sample(rssm.get_feat(state))
        state = rssm.img_step(state, act)
        feats.append(rssm.get_feat(state))

    feats_t = torch.stack(feats, dim=0)
    assert feats_t.shape == (H + 1, M, deter_dim + stoch_dim)


# ──────────────────────────────────────────────────────────────────────
# SequenceBuffer
# ──────────────────────────────────────────────────────────────────────


def _make_buffer_with_episodes(
    n_episodes: int = 3,
    ep_len: int = 20,
    obs_dim: int = 6,
    action_dim: int = 3,
    capacity: int = 10000,
) -> SequenceBuffer:
    rng = np.random.default_rng(42)
    buf = SequenceBuffer(capacity)
    for _ in range(n_episodes):
        for step in range(ep_len):
            obs = rng.standard_normal(obs_dim).astype(np.float32)
            action = rng.standard_normal(action_dim).astype(np.float32)
            reward = float(rng.standard_normal())
            done = step == ep_len - 1
            buf.add(obs, action, reward, done)
    return buf


def test_sequence_buffer_len() -> None:
    """Buffer len matches total stored transitions."""
    ep_len, n_ep = 20, 3
    buf = _make_buffer_with_episodes(n_ep, ep_len)
    assert len(buf) == ep_len * n_ep


def test_sequence_buffer_sample_shapes() -> None:
    """sample(B, L) returns float32 tensors with shapes [B, L, ...]."""
    obs_dim, action_dim = 6, 3
    buf = _make_buffer_with_episodes(n_episodes=4, ep_len=25, obs_dim=obs_dim, action_dim=action_dim)

    B, L = 4, 10
    batch = buf.sample(B, L)

    assert batch["obs"].shape == (B, L, obs_dim)
    assert batch["action"].shape == (B, L, action_dim)
    assert batch["reward"].shape == (B, L)
    assert batch["done"].shape == (B, L)
    assert batch["obs"].dtype == torch.float32
    assert batch["action"].dtype == torch.float32
    assert batch["reward"].dtype == torch.float32


def test_sequence_buffer_raises_if_no_eligible_episode() -> None:
    """Requesting seq_len longer than any episode raises RuntimeError."""
    buf = _make_buffer_with_episodes(n_episodes=2, ep_len=5)
    with pytest.raises(RuntimeError):
        buf.sample(2, 100)


def test_sequence_buffer_flush_current() -> None:
    """flush_current seals an in-progress episode so it can be sampled."""
    obs_dim, action_dim = 4, 2
    buf = SequenceBuffer(1000)
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        action = rng.standard_normal(action_dim).astype(np.float32)
        buf.add(obs, action, 0.0, False)   # done=False → not auto-flushed

    assert len(buf) == 10   # transitions are in _current
    buf.flush_current()
    # Now sealed; can be sampled
    batch = buf.sample(1, 5)
    assert batch["obs"].shape == (1, 5, obs_dim)


def test_sequence_buffer_capacity_enforcement() -> None:
    """Old episodes are dropped when capacity is exceeded."""
    buf = SequenceBuffer(capacity=50)
    rng = np.random.default_rng(1)

    for _ in range(10):  # 10 episodes × 10 steps = 100 transitions > 50
        for step in range(10):
            obs = rng.standard_normal(4).astype(np.float32)
            buf.add(obs, np.zeros(2, np.float32), 0.0, done=(step == 9))

    assert len(buf) <= 50 + 10  # at most one extra episode


# ======================================================================
# Part 2 — Integration tests
# ======================================================================


from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY  # noqa: E402
from rl_from_scratch.dreamer.agent import DreamerAgent  # noqa: E402
from rl_from_scratch.dreamer.config import DreamerConfig  # noqa: E402
from rl_from_scratch.dreamer.training import evaluate, train_dreamer  # noqa: E402


def _disable_dreamer_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_from_scratch.dreamer.reporting as _rep
    monkeypatch.setattr(_rep, "generate_training_figures", lambda *a, **kw: [])


def _make_dreamer_agent(
    obs_dim: int = 17,
    action_dim: int = 6,
    seed: int = 0,
) -> DreamerAgent:
    torch.manual_seed(seed)
    return DreamerAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=np.full(action_dim, -1.0, dtype=np.float32),
        action_high=np.full(action_dim, 1.0, dtype=np.float32),
        deter_dim=16,
        stoch_dim=8,
        rssm_hidden_dim=16,
        encoder_hidden_dim=16,
        decoder_hidden_dim=16,
        reward_hidden_dim=16,
        embed_dim=16,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        batch_size=4,
        batch_length=5,
        imagination_horizon=3,
        num_warmup_steps=5,
        buffer_capacity=10000,
        device="cpu",
    )


def _fill_agent_buffer(agent: DreamerAgent, n_episodes: int = 4, ep_len: int = 10) -> None:
    """Fill the agent's buffer with toy episodes."""
    rng = np.random.default_rng(42)
    for _ in range(n_episodes):
        for step in range(ep_len):
            obs = rng.standard_normal(agent.obs_dim).astype(np.float32)
            action = rng.standard_normal(agent.action_dim).astype(np.float32)
            reward = float(rng.standard_normal())
            done = step == ep_len - 1
            agent.buffer.add(obs, action, reward, done)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_config_registered() -> None:
    assert "dreamer" in CONFIG_REGISTRY
    assert CONFIG_REGISTRY["dreamer"] is DreamerConfig


def test_dreamer_agent_factory_registered() -> None:
    assert "dreamer" in AGENT_FACTORIES
    assert AGENT_FACTORIES["dreamer"] is train_dreamer


# ──────────────────────────────────────────────────────────────────────
# Config validation
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_config_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_"):
        DreamerConfig(lambda_=0.0)


def test_dreamer_config_rejects_bad_imagination_horizon() -> None:
    with pytest.raises(ValueError, match="imagination_horizon"):
        DreamerConfig(imagination_horizon=0)


def test_dreamer_config_rejects_bad_deter_dim() -> None:
    with pytest.raises(ValueError, match="deter_dim"):
        DreamerConfig(deter_dim=0)


def test_dreamer_config_rejects_bad_stoch_dim() -> None:
    with pytest.raises(ValueError, match="stoch_dim"):
        DreamerConfig(stoch_dim=-1)


def test_dreamer_config_rejects_bad_batch_length() -> None:
    with pytest.raises(ValueError, match="batch_length"):
        DreamerConfig(batch_length=1)


def test_dreamer_config_rejects_bad_train_every() -> None:
    with pytest.raises(ValueError, match="train_every"):
        DreamerConfig(train_every=0)


def test_dreamer_config_round_trips_through_dict() -> None:
    config = DreamerConfig(epochs=5, deter_dim=64, imagination_horizon=10)
    d = config.to_dict()
    restored = DreamerConfig.from_dict(d)
    assert restored.epochs == 5
    assert restored.deter_dim == 64
    assert restored.imagination_horizon == 10


# ──────────────────────────────────────────────────────────────────────
# select_action shape/dtype/bounds
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_select_action_during_warmup() -> None:
    """During warm-up, select_action returns a random action in [low, high]."""
    agent = _make_dreamer_agent(obs_dim=17, action_dim=6)
    obs = np.zeros(17, dtype=np.float32)
    # Buffer empty → warm-up mode
    action = agent.select_action(obs)
    assert action.shape == (6,), f"Expected shape (6,), got {action.shape}"
    assert action.dtype == np.float32
    assert np.all(action >= -1.0 - 1e-5)
    assert np.all(action <= 1.0 + 1e-5)


def test_dreamer_select_action_after_warmup() -> None:
    """After warm-up, select_action uses the actor and stays in bounds."""
    agent = _make_dreamer_agent(obs_dim=17, action_dim=6)
    _fill_agent_buffer(agent, n_episodes=4, ep_len=10)  # > num_warmup_steps=5

    obs = np.zeros(17, dtype=np.float32)
    action = agent.select_action(obs, deterministic=False)
    assert action.shape == (6,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0 - 1e-5)
    assert np.all(action <= 1.0 + 1e-5)


def test_dreamer_deterministic_action_reproducible_after_recurrent_reset() -> None:
    """Deterministic evaluation must use the actor mean, not sampling noise."""
    torch.manual_seed(0)
    np.random.seed(0)
    agent = _make_dreamer_agent(obs_dim=3, action_dim=1)
    _fill_agent_buffer(agent, n_episodes=4, ep_len=10)
    obs = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    agent.episode_ended()
    action_1 = agent.select_action(obs, deterministic=True)
    agent.episode_ended()
    action_2 = agent.select_action(obs, deterministic=True)

    np.testing.assert_allclose(action_1, action_2, atol=1e-6)


def test_dreamer_evaluate_restores_training_recurrent_state() -> None:
    """Greedy eval may advance RSSM internally but must not disturb training state."""
    agent = _make_dreamer_agent(obs_dim=3, action_dim=1)
    _fill_agent_buffer(agent, n_episodes=4, ep_len=10)
    obs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    agent.select_action(obs, deterministic=True)

    saved_state = (
        agent._h.detach().clone(),
        agent._z.detach().clone(),
        agent._prev_action.detach().clone(),
    )

    summary = evaluate(
        agent,
        "Pendulum-v1",
        n_episodes=1,
        seed=123,
        max_steps=3,
    )

    assert "mean_reward" in summary
    assert agent._h is not None and agent._z is not None and agent._prev_action is not None
    torch.testing.assert_close(agent._h, saved_state[0])
    torch.testing.assert_close(agent._z, saved_state[1])
    torch.testing.assert_close(agent._prev_action, saved_state[2])


# ──────────────────────────────────────────────────────────────────────
# learn_step
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_learn_step_returns_finite_metrics() -> None:
    """learn_step returns finite float metrics when buffer has enough data."""
    torch.manual_seed(0)
    agent = _make_dreamer_agent()
    # batch_size=4, batch_length=5 → need ≥ 20 transitions + at least one episode ≥ 5
    _fill_agent_buffer(agent, n_episodes=6, ep_len=10)

    metrics = agent.learn_step()
    expected_keys = {
        "recon_loss", "reward_loss", "kl", "model_loss",
        "actor_loss", "critic_loss", "imagined_return",
    }
    assert expected_keys == set(metrics.keys()), (
        f"Missing keys: {expected_keys - set(metrics.keys())}"
    )
    for k, v in metrics.items():
        assert math.isfinite(float(v)), f"Metric {k} is not finite: {v}"


def test_dreamer_learn_step_zeros_when_buffer_too_small() -> None:
    """learn_step returns all-zero dict when buffer is too small."""
    agent = _make_dreamer_agent()
    metrics = agent.learn_step()
    # Should return the zero dict (not raise)
    assert isinstance(metrics, dict)
    for v in metrics.values():
        assert float(v) == 0.0


# ──────────────────────────────────────────────────────────────────────
# save / load round-trip
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_save_load_action_matches(tmp_path: Path) -> None:
    """After save/load, deterministic action matches original to atol=1e-5.

    We test the *deterministic* mode (tanh(mean)), which is fully reproducible
    without depending on random sampling.  Both agents start from the same
    zero-initialised recurrent state (episode_ended resets it).
    """
    torch.manual_seed(3)
    agent = _make_dreamer_agent(seed=3)
    _fill_agent_buffer(agent, n_episodes=4, ep_len=10)

    ckpt_path = agent.save(tmp_path / "dreamer.pt")
    loaded = DreamerAgent.load(ckpt_path, device="cpu")

    # Fill loaded agent's buffer so it passes warm-up check
    _fill_agent_buffer(loaded, n_episodes=4, ep_len=10)

    obs = np.zeros(17, dtype=np.float32)

    # Reset recurrent state in both agents so they both start from the same h=0
    agent.episode_ended()
    loaded.episode_ended()

    # Deterministic mode uses tanh(mean), no stochastic sampling
    action_orig = agent.select_action(obs, deterministic=True)
    action_loaded = loaded.select_action(obs, deterministic=True)

    np.testing.assert_allclose(
        action_orig, action_loaded, atol=1e-5,
        err_msg="Deterministic action should match after save/load round-trip",
    )


# ──────────────────────────────────────────────────────────────────────
# Smoke training test
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_dreamer with a tiny config completes and returns the expected contract."""
    pytest.importorskip(
        "gymnasium.envs.mujoco",
        reason="MuJoCo not available; skipping Dreamer smoke test",
    )

    _disable_dreamer_figures(monkeypatch)

    try:
        config = DreamerConfig(
            env_id="HalfCheetah-v5",
            deter_dim=16,
            stoch_dim=8,
            rssm_hidden_dim=16,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            reward_hidden_dim=16,
            embed_dim=16,
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            batch_size=4,
            batch_length=5,
            imagination_horizon=3,
            epochs=1,
            steps_per_epoch=10,
            max_steps_per_episode=10,
            num_warmup_steps=10,
            eval_every=1,
            eval_episodes=1,
            checkpoint_every=1,
            output_dir=str(tmp_path),
        )

        result = train_dreamer(config, seed=0)

        assert set(result) == {"agent", "history", "metrics", "paths"}
        assert isinstance(result["agent"], DreamerAgent)
        assert isinstance(result["history"], dict)
        assert result["paths"].run_dir.exists()

    except Exception as exc:
        err_msg = str(exc).lower()
        if "mujoco" in err_msg or "halfcheetah" in err_msg:
            pytest.skip(f"MuJoCo/HalfCheetah unavailable: {exc}")
        raise


# ──────────────────────────────────────────────────────────────────────
# Cross-package import isolation (AST walk)
# ──────────────────────────────────────────────────────────────────────


def test_dreamer_has_no_cross_package_imports() -> None:
    """dreamer package must not import from sibling algorithm packages."""
    dreamer_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rl_from_scratch"
        / "dreamer"
    )

    forbidden = {
        "rl_from_scratch.sac",
        "rl_from_scratch.mbpo",
        "rl_from_scratch.pets",
        "rl_from_scratch.pilco",
        "rl_from_scratch.deep_q",
        "rl_from_scratch.actor_critic",
        "rl_from_scratch.deterministic_actor_critic",
        "rl_from_scratch.trust_region",
        "rl_from_scratch.reinforce",
        "rl_from_scratch.tabular",
        "rl_from_scratch.dyna",
    }

    for path in sorted(dreamer_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
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
