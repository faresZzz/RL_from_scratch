"""Tests for the autonomous trust_region package (TRPO and PPO)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_from_scratch.core.base import BaseAgent
from rl_from_scratch.core.config import load_config
from rl_from_scratch.trust_region.agent import PPOAgent, TRPOAgent
from rl_from_scratch.trust_region.buffer import TrustRegionRolloutBuffer
from rl_from_scratch.trust_region.config import PPOConfig, TRPOConfig
from rl_from_scratch.trust_region.training import train_ppo, train_trpo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUST_REGION_DIR = PROJECT_ROOT / "src/rl_from_scratch/trust_region"


def _assert_float_metrics(result: dict[str, float], expected_keys: set[str]) -> None:
    assert expected_keys.issubset(result.keys()), (
        f"Missing expected keys: {expected_keys - set(result.keys())}"
    )
    for key in expected_keys:
        value = result[key]
        assert isinstance(value, float), f"{key} should be a float, got {type(value)}"
        assert np.isfinite(value), f"{key} should be finite, got {value}"


def _make_trpo_agent(
    obs_dim: int = 3,
    action_dim: int = 1,
    n_steps: int = 32,
    **kwargs: object,
) -> TRPOAgent:
    return TRPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_steps=n_steps,
        hidden_dim=32,
        device="cpu",
        **kwargs,
    )


def test_trust_region_configs_load_smoke_and_full_variants() -> None:
    config_dir = PROJECT_ROOT / "configs" / "trust_region"

    trpo_smoke = load_config(config_dir / "trpo_pendulum_smoke.yaml")
    trpo_full = load_config(config_dir / "trpo_halfcheetah.yaml")
    ppo_smoke = load_config(config_dir / "ppo_pendulum_smoke.yaml")
    ppo_full = load_config(config_dir / "ppo_halfcheetah.yaml")

    assert isinstance(trpo_smoke, TRPOConfig)
    assert trpo_smoke.env_id == "Pendulum-v1"
    assert trpo_smoke.total_timesteps == 256
    assert trpo_smoke.backtrack_iters == 5
    assert isinstance(trpo_full, TRPOConfig)
    assert trpo_full.env_id == "HalfCheetah-v5"
    assert trpo_full.normalize_observations is True

    assert isinstance(ppo_smoke, PPOConfig)
    assert ppo_smoke.env_id == "Pendulum-v1"
    assert ppo_smoke.total_timesteps == 256
    assert ppo_smoke.n_epochs == 2
    assert isinstance(ppo_full, PPOConfig)
    assert ppo_full.env_id == "HalfCheetah-v5"
    assert ppo_full.normalize_observations is True


def test_trust_region_configs_reject_legacy_line_search_keys() -> None:
    with pytest.raises(ValueError, match="Unknown config keys.*line_search_steps"):
        TRPOConfig.from_dict({"approach": "trpo", "line_search_steps": 10})


def _make_ppo_agent(
    obs_dim: int = 3,
    action_dim: int = 1,
    n_steps: int = 32,
    **kwargs: object,
) -> PPOAgent:
    kwargs.setdefault("batch_size", 16)
    return PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_steps=n_steps,
        hidden_dim=32,
        device="cpu",
        **kwargs,
    )


def _fill_agent_buffer(agent: BaseAgent, n_steps: int, obs_dim: int) -> None:
    obs = np.random.randn(obs_dim).astype(np.float32)
    for _ in range(n_steps):
        action = agent.select_action(obs)
        agent.store_transition(obs, action, 1.0, obs, False)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_trust_region_package_owns_its_core_modules() -> None:
    expected = {"agent.py", "buffer.py", "config.py", "network.py", "training.py"}
    actual = {path.name for path in TRUST_REGION_DIR.glob("*.py")}
    assert expected.issubset(actual)
    assert "rollout.py" not in actual


def test_trust_region_training_owns_rollout_helpers() -> None:
    training_path = TRUST_REGION_DIR / "training.py"
    tree = ast.parse(training_path.read_text(encoding="utf-8"), filename=str(training_path))
    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "train_one_episode" in function_names
    # evaluate is now either in core/evaluate.py (generic) or a local closure
    # (tr_evaluate) inside _train_trust_region for obs-normalised evaluation.
    assert "TrustRegionRunner" not in training_path.read_text(encoding="utf-8")


def test_trust_region_modules_do_not_import_sibling_algorithm_packages() -> None:
    offending: dict[str, list[str]] = {}
    for path in TRUST_REGION_DIR.glob("*.py"):
        bad = sorted(
            name
            for name in _module_imports(path)
            if name.startswith("rl_from_scratch.actor_critic")
            or name.startswith("rl_from_scratch.deterministic_actor_critic")
            or name.startswith("rl_from_scratch.deep_q")
            or name.startswith("rl_from_scratch.reinforce")
            or name.startswith("rl_from_scratch.tabular")
        )
        if bad:
            offending[path.name] = bad

    assert offending == {}, f"Sibling algorithm imports remain: {offending}"


def test_trpo_and_ppo_are_autonomous_base_agents() -> None:
    trpo = _make_trpo_agent()
    ppo = _make_ppo_agent()

    assert isinstance(trpo, BaseAgent)
    assert isinstance(ppo, BaseAgent)
    assert trpo.__class__.__mro__[1] is not PPOAgent
    assert "actor_critic" not in trpo.__class__.__module__
    assert "actor_critic" not in ppo.__class__.__module__
    assert trpo.buffer.__class__.__module__ == "rl_from_scratch.trust_region.buffer"
    assert ppo.buffer.__class__.__module__ == "rl_from_scratch.trust_region.buffer"
    assert trpo.actor.__class__.__module__ == "rl_from_scratch.trust_region.network"
    assert ppo.actor.__class__.__module__ == "rl_from_scratch.trust_region.network"


def test_trust_region_buffer_computes_gae_returns() -> None:
    buffer = TrustRegionRolloutBuffer(n_steps=4, obs_dim=3, action_dim=1)
    for _ in range(4):
        buffer.push(
            np.ones(3, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1.0,
            False,
            -0.5,
            0.25,
        )

    returns, advantages = buffer.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert returns.shape == (4,)
    assert advantages.shape == (4,)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()


def test_trpo_fisher_vector_product_shape() -> None:
    agent = _make_trpo_agent(obs_dim=4, action_dim=2)
    obs = torch.randn(16, 4)

    with torch.no_grad():
        old_dist = agent.actor.get_distribution(obs)

    n_params = sum(p.numel() for p in agent.actor.parameters())
    vector = torch.randn(n_params)

    fvp = agent._fisher_vector_product(vector, obs, old_dist)
    assert fvp.shape == (n_params,)


def test_trpo_conjugate_gradient_solves_linear_system() -> None:
    n = 20
    diagonal = torch.ones(n) + 0.1 * torch.rand(n)

    def av_fn(vector: torch.Tensor) -> torch.Tensor:
        return diagonal * vector

    rhs = torch.randn(n)
    agent = _make_trpo_agent()
    solution = agent._conjugate_gradient(av_fn, rhs, n_steps=50)

    residual = (diagonal * solution - rhs).norm().item()
    assert residual < 1e-3


def test_trpo_update_respects_kl_constraint() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    max_kl = 0.1
    agent = _make_trpo_agent(obs_dim=3, action_dim=1, n_steps=64, max_kl=max_kl)

    obs = np.random.randn(3).astype(np.float32)
    for _ in range(64):
        action = agent.select_action(obs)
        agent.store_transition(obs, action, 1.0, obs, False)

    returns, advantages = agent.buffer.compute_gae(
        next_value=0.0,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
    )
    batch = agent.buffer.get_batch(device=agent.device)

    with torch.no_grad():
        old_dist = agent.actor.get_distribution(batch["obs"])

    agent._update(advantages.to(agent.device), returns.to(agent.device), batch)

    with torch.no_grad():
        new_dist = agent.actor.get_distribution(batch["obs"])
        kl = torch.distributions.kl_divergence(old_dist, new_dist)
        kl_mean = kl.sum(dim=-1).mean().item()

    assert kl_mean <= max_kl + 1e-6


def test_trpo_learn_step_returns_autonomous_metrics() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    agent = _make_trpo_agent(n_steps=32)
    _fill_agent_buffer(agent, 32, obs_dim=3)

    result = agent.learn_step(next_value=0.0)

    expected_keys = {
        "policy_loss",
        "value_loss",
        "kl",
        "entropy",
        "actual_kl",
        "line_search_accept",
        "line_search_step_fraction",
        "adv_mean",
        "adv_std",
        "explained_variance",
        "grad_norm",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
        "action_abs_mean",
        "action_clip_fraction",
    }
    _assert_float_metrics(result, expected_keys)


def test_ppo_old_log_probs_match_stored_policy_actions_before_update() -> None:
    torch.manual_seed(123)
    np.random.seed(123)

    agent = _make_ppo_agent(n_steps=32, n_epochs=1)
    _fill_agent_buffer(agent, 32, obs_dim=3)
    batch = agent.buffer.get_batch(device=agent.device)

    with torch.no_grad():
        dist = agent.actor.get_distribution(batch["obs"])
        new_log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)
        ratio = torch.exp(new_log_probs - batch["log_probs"])

    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_ppo_early_stopping_on_kl_reports_metrics() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    agent = _make_ppo_agent(n_steps=64, n_epochs=10, target_kl=1e-6, batch_size=32)
    _fill_agent_buffer(agent, 64, obs_dim=3)

    returns, advantages = agent.buffer.compute_gae(
        next_value=0.0,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
    )
    batch = agent.buffer.get_batch(device=agent.device)

    result = agent._update(advantages.to(agent.device), returns.to(agent.device), batch)

    assert "kl" in result
    assert "policy_loss" in result
    assert result["kl"] >= 0.0


def test_ppo_learn_step_returns_autonomous_metrics() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    agent = _make_ppo_agent(n_steps=32)
    _fill_agent_buffer(agent, 32, obs_dim=3)

    result = agent.learn_step(next_value=0.0)

    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "kl",
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "ratio_std",
        "adv_mean",
        "adv_std",
        "explained_variance",
        "grad_norm",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
        "action_abs_mean",
        "action_clip_fraction",
    }
    _assert_float_metrics(result, expected_keys)


def test_trpo_training_smoke(tmp_path) -> None:
    config = TRPOConfig(
        env_id="Pendulum-v1",
        total_timesteps=4096,
        n_steps=256,
        hidden_dim=32,
        checkpoint_every=4096,
        value_train_iters=10,
        cg_iters=5,
        device="cpu",
    )
    result = train_trpo(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], TRPOAgent)


def test_ppo_training_smoke(tmp_path) -> None:
    config = PPOConfig(
        env_id="Pendulum-v1",
        total_timesteps=4096,
        n_steps=256,
        hidden_dim=32,
        checkpoint_every=4096,
        n_epochs=4,
        batch_size=64,
        device="cpu",
    )
    result = train_ppo(config, output_dir=str(tmp_path), seed=0)

    assert "agent" in result
    assert "history" in result
    assert "metrics" in result
    assert "paths" in result
    assert isinstance(result["agent"], PPOAgent)
