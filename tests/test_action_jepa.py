"""Tests for the autonomous Action-JEPA package."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import rl_from_scratch  # noqa: F401

from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY
from rl_from_scratch.action_jepa.agent import ActionJepaAgent
from rl_from_scratch.action_jepa.buffer import SequenceBuffer
from rl_from_scratch.action_jepa.config import ActionJepaConfig
from rl_from_scratch.action_jepa.networks import (
    ContinuationHead,
    Encoder,
    MaskedContextPredictor,
    Predictor,
    RewardHead,
    covariance_loss,
    ema_update,
    latent_collapse_metric,
    variance_loss,
)
from rl_from_scratch.action_jepa.planner import (
    LatentCEMPlanner,
    goal_objective,
    reward_objective,
)
from rl_from_scratch.action_jepa.training import evaluate, train_action_jepa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src/rl_from_scratch/action_jepa"


@pytest.fixture(autouse=True)
def _float32():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


def _make_agent(**overrides: object) -> ActionJepaAgent:
    base = dict(
        obs_dim=3,
        action_dim=1,
        action_low=np.array([-2.0], dtype=np.float32),
        action_high=np.array([2.0], dtype=np.float32),
        latent_dim=8,
        hidden_dim=32,
        encoder_layers=2,
        ema_tau=0.9,
        rollout_len=3,
        batch_size=4,
        buffer_capacity=200,
        learning_starts=8,
        num_warmup_steps=8,
        plan_horizon=4,
        cem_population=64,
        cem_num_elites=8,
        cem_iterations=3,
        cem_alpha=0.5,
        device="cpu",
        seed=0,
        plan_mode="goal",
        goal_obs=[1.0, 0.0, 0.0],
    )
    base.update(overrides)
    return ActionJepaAgent(**base)


def _fill_agent_buffer(agent: ActionJepaAgent, steps: int = 18) -> None:
    obs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for idx in range(steps):
        action = np.array([math.sin(idx)], dtype=np.float32)
        next_obs = np.array(
            [
                math.cos(0.1 * (idx + 1)),
                math.sin(0.1 * (idx + 1)),
                0.1 * (idx + 1),
            ],
            dtype=np.float32,
        )
        reward = float(1.0 - abs(next_obs[1]))
        done = (idx + 1) % 6 == 0
        agent.store_transition(obs, action, reward, next_obs, done)
        obs = np.array([1.0, 0.0, 0.0], dtype=np.float32) if done else next_obs
    agent.episode_ended()


def _disable_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_from_scratch.action_jepa.reporting as _action_jepa_reporting
    import rl_from_scratch.core.reporting as _core_reporting

    monkeypatch.setattr(_action_jepa_reporting, "generate_training_figures", lambda *a, **k: [])
    monkeypatch.setattr(_core_reporting, "record_greedy_episode", lambda *a, **k: None)


def test_network_shapes() -> None:
    encoder = Encoder(obs_dim=3, latent_dim=8, hidden_dim=16, n_layers=2)
    predictor = Predictor(latent_dim=8, action_dim=1, hidden_dim=16, delta=True)
    reward = RewardHead(latent_dim=8, action_dim=1, hidden_dim=16)
    cont = ContinuationHead(latent_dim=8, action_dim=1, hidden_dim=16)

    obs = torch.randn(5, 3)
    latents = encoder(obs)
    next_latents = predictor(latents, torch.randn(5, 1))
    reward_pred = reward(latents, torch.randn(5, 1))
    cont_logits = cont(latents, torch.randn(5, 1))

    assert latents.shape == (5, 8)
    assert next_latents.shape == (5, 8)
    assert reward_pred.shape == (5,)
    assert cont_logits.shape == (5,)


def test_predictor_delta_adds_residual() -> None:
    torch.manual_seed(0)
    predictor = Predictor(latent_dim=4, action_dim=2, hidden_dim=8, delta=True)
    predictor_delta_off = Predictor(latent_dim=4, action_dim=2, hidden_dim=8, delta=False)
    predictor_delta_off.load_state_dict(predictor.state_dict())

    z = torch.randn(3, 4)
    a = torch.randn(3, 2)

    residual = predictor.net(torch.cat([z, a], dim=-1))
    out_delta = predictor(z, a)
    out_plain = predictor_delta_off(z, a)

    assert torch.allclose(out_delta, z + residual, atol=1e-6)
    assert torch.allclose(out_plain, residual, atol=1e-6)


def test_ema_update_moves_target_toward_online() -> None:
    online = torch.nn.Linear(4, 4)
    target = torch.nn.Linear(4, 4)

    with torch.no_grad():
        for param in online.parameters():
            param.fill_(1.0)
        for param in target.parameters():
            param.zero_()

    ema_update(target, online, tau=0.5)
    target_weight = next(target.parameters()).detach()
    assert torch.allclose(target_weight, torch.full_like(target_weight, 0.5))


def test_vicreg_losses_behave_sensibly() -> None:
    constant = torch.ones(16, 6)
    varied = torch.randn(32, 6)

    assert variance_loss(constant, target_std=1.0) > 0.5
    assert variance_loss(varied, target_std=0.1) < 1e-3
    assert covariance_loss(varied) >= 0.0
    assert latent_collapse_metric(varied) > latent_collapse_metric(constant)


def test_sequence_buffer_samples_contiguous_seedable_windows() -> None:
    buffer_a = SequenceBuffer(capacity=100, seed=123)
    buffer_b = SequenceBuffer(capacity=100, seed=123)

    for buffer in (buffer_a, buffer_b):
        for episode_id in range(2):
            for step in range(5):
                obs = np.array([episode_id, step], dtype=np.float32)
                action = np.array([step], dtype=np.float32)
                next_obs = np.array([episode_id, step + 1], dtype=np.float32)
                buffer.add(obs, action, float(step), next_obs, done=step == 4)

    batch_a = buffer_a.sample(batch_size=3, rollout_len=2)
    batch_b = buffer_b.sample(batch_size=3, rollout_len=2)

    for key in ("obs", "action", "reward", "done", "next_obs"):
        assert torch.allclose(batch_a[key], batch_b[key])

    obs = batch_a["obs"].numpy()
    next_obs = batch_a["next_obs"].numpy()
    assert obs.shape == (3, 3, 2)
    assert np.all(obs[:, :, 0] == obs[:, :1, 0])
    assert np.allclose(next_obs[:, :-1, 1], obs[:, 1:, 1])


def test_planner_goal_objective_respects_bounds_and_improves_distance() -> None:
    planner = LatentCEMPlanner(
        action_dim=1,
        horizon=4,
        population=128,
        num_elites=16,
        iterations=4,
        alpha=0.5,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        seed=0,
    )

    def predictor(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + action

    z0 = torch.zeros(1)
    z_goal = torch.full((1,), 0.75)
    objective = goal_objective(z_goal)
    action, mean = planner.plan(z0, predictor, objective=objective)

    assert action.shape == (1,)
    assert -1.0 <= float(action[0]) <= 1.0
    assert mean.shape == (4, 1)

    planned_latent = predictor(z0.unsqueeze(0), torch.tensor(action).unsqueeze(0)).squeeze(0)
    random_latent = predictor(z0.unsqueeze(0), torch.tensor([[0.0]])).squeeze(0)
    assert torch.norm(planned_latent - z_goal) < torch.norm(random_latent - z_goal)


def test_planner_reward_objective_prefers_higher_predicted_return() -> None:
    planner = LatentCEMPlanner(
        action_dim=1,
        horizon=3,
        population=96,
        num_elites=12,
        iterations=4,
        alpha=0.5,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        seed=1,
    )

    def predictor(z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + 0.5 * action

    class ToyReward(torch.nn.Module):
        def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return 2.0 * action.squeeze(-1)

    class ToyCont(torch.nn.Module):
        def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return torch.full((z.shape[0],), 8.0)

    objective = reward_objective(ToyReward(), ToyCont(), gamma=0.99)
    action, _ = planner.plan(torch.zeros(1), predictor, objective=objective)
    assert float(action[0]) > 0.25


def test_reward_objective_scores_from_pre_action_latents() -> None:
    class StateReward(torch.nn.Module):
        def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            del action
            return z.squeeze(-1)

    class AlwaysContinue(torch.nn.Module):
        def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            del action
            return torch.full((z.shape[0],), 8.0)

    objective = reward_objective(StateReward(), AlwaysContinue(), gamma=1.0)
    rollout_latents = torch.tensor([[[2.0], [5.0]]])
    actions = torch.zeros(1, 2, 1)
    score = objective(rollout_latents, actions, torch.tensor([1.0]))

    # Reward must use z_t before each action: [z0=1, z1=2], not [z1=2, z2=5].
    # Compounding survival: r(z0) at survival=1, then r(z1) weighted by the
    # prior continuation probability sigmoid(8).
    c = torch.sigmoid(torch.tensor(8.0))
    expected = 1.0 + c * 2.0
    assert score.item() == pytest.approx(expected.item(), rel=1e-5)


def test_registry_contains_action_jepa() -> None:
    assert "action_jepa" in CONFIG_REGISTRY
    assert "action_jepa" in AGENT_FACTORIES


def test_config_validation_and_roundtrip() -> None:
    config = ActionJepaConfig()
    payload = config.to_dict()
    restored = ActionJepaConfig.from_dict(payload)

    assert restored.approach == "action_jepa"
    assert restored.plan_mode == config.plan_mode

    with pytest.raises(ValueError):
        ActionJepaConfig(ema_tau=1.5)
    with pytest.raises(ValueError):
        ActionJepaConfig(rollout_len=0)
    with pytest.raises(ValueError):
        ActionJepaConfig(plan_mode="mystery")
    with pytest.raises(ValueError, match="training_regime"):
        ActionJepaConfig(training_regime="mystery")
    with pytest.raises(ValueError, match="freeze_encoder_after_pretrain"):
        ActionJepaConfig(training_regime="stage-wise")
    with pytest.raises(ValueError, match="random_frozen_encoder"):
        ActionJepaConfig(training_regime="random-frozen", random_frozen_encoder=False)


def test_masked_context_predictor_shapes() -> None:
    predictor = MaskedContextPredictor(latent_dim=8, obs_dim=3, hidden_dim=16)
    context = torch.randn(5, 8)
    partial = torch.randn(5, 8)
    mask = torch.ones(5, 3)
    output = predictor(context, partial, mask)
    assert output.shape == (5, 8)


def test_select_action_stays_within_bounds() -> None:
    agent = _make_agent()
    action = agent.select_action(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert action.shape == (1,)
    assert -2.0 <= float(action[0]) <= 2.0


def test_learn_step_returns_finite_metrics_and_changes_params() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    agent = _make_agent()
    _fill_agent_buffer(agent)

    before = [param.detach().clone() for param in agent.encoder.parameters()]
    metrics = agent.learn_step()

    assert metrics
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert "latent_prediction_loss" in metrics
    assert "rollout_prediction_loss" in metrics
    assert "reward_prediction_loss" in metrics
    assert metrics["latent_std"] > 0.0
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, agent.encoder.parameters())
    )


def test_representation_step_updates_encoder_and_reports_health_metrics() -> None:
    torch.manual_seed(0)
    agent = _make_agent(training_regime="stage-wise", freeze_encoder_after_pretrain=True)
    _fill_agent_buffer(agent)

    before = [param.detach().clone() for param in agent.encoder.parameters()]
    metrics = agent.representation_step()

    assert metrics["representation_prediction_loss"] >= 0.0
    assert metrics["latent_std"] > 0.0
    assert metrics["effective_rank"] > 0.0
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, agent.encoder.parameters())
    )


def test_frozen_encoder_is_not_updated_by_action_conditioned_step() -> None:
    torch.manual_seed(0)
    agent = _make_agent(training_regime="stage-wise", freeze_encoder_after_pretrain=True)
    _fill_agent_buffer(agent)
    agent.representation_step()
    agent.freeze_encoder()

    before_encoder = [param.detach().clone() for param in agent.encoder.parameters()]
    before_predictor = [param.detach().clone() for param in agent.predictor.parameters()]
    metrics = agent.learn_step()

    assert metrics["total_loss"] >= 0.0
    assert all(
        torch.allclose(old, new)
        for old, new in zip(before_encoder, agent.encoder.parameters())
    )
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before_predictor, agent.predictor.parameters())
    )


def test_random_frozen_starts_with_frozen_encoder() -> None:
    agent = _make_agent(training_regime="random-frozen", random_frozen_encoder=True)
    assert agent.encoder_frozen is True
    assert all(not parameter.requires_grad for parameter in agent.encoder.parameters())


def test_save_load_roundtrip(tmp_path: Path) -> None:
    agent = _make_agent()
    _fill_agent_buffer(agent, steps=12)
    agent.learn_step()

    checkpoint = agent.save(tmp_path / "action_jepa.pt")
    loaded = ActionJepaAgent.load(checkpoint, device="cpu")

    obs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    action_a = agent.select_action(obs, deterministic=True)
    action_b = loaded.select_action(obs, deterministic=True)
    assert np.allclose(action_a, action_b, atol=1e-5)


def test_evaluate_restores_planner_state() -> None:
    agent = _make_agent(cem_population=16, cem_num_elites=4, cem_iterations=1)
    agent._collected_steps = agent.num_warmup_steps
    agent._prev_mean = torch.ones(agent.plan_horizon, agent.action_dim) * 0.25
    saved_mean = agent._prev_mean.clone()
    saved_rng = agent.planner.get_rng_state().clone()

    result = evaluate(
        agent,
        "Pendulum-v1",
        n_episodes=1,
        seed=123,
        max_steps=1,
    )

    assert math.isfinite(result["mean_reward"])
    assert torch.allclose(agent._prev_mean, saved_mean)
    assert torch.equal(agent.planner.get_rng_state(), saved_rng)


def test_train_action_jepa_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _disable_figures(monkeypatch)
    config = ActionJepaConfig(
        env_id="Pendulum-v1",
        seed=0,
        output_dir=str(tmp_path),
        run_name="smoke",
        total_timesteps=128,
        episodes=1,
        max_steps_per_episode=20,
        eval_every=1,
        eval_episodes=1,
        checkpoint_every=1,
        num_warmup_steps=8,
        learning_starts=8,
        pretrain_steps=4,
        collect_every=2,
        updates_per_collect=1,
        batch_size=4,
        rollout_len=2,
        plan_horizon=3,
        cem_population=32,
        cem_num_elites=4,
        cem_iterations=2,
        latent_dim=8,
        hidden_dim=32,
        plan_mode="goal",
        goal_obs=[1.0, 0.0, 0.0],
        device="cpu",
    )

    result = train_action_jepa(config)

    assert {"agent", "history", "metrics", "paths"} <= set(result)
    assert result["history"]["episode_rewards"]


def test_action_jepa_package_does_not_import_sibling_algorithms() -> None:
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("rl_from_scratch.") and not (
                    node.module.startswith("rl_from_scratch.core")
                    or node.module.startswith("rl_from_scratch.action_jepa")
                ):
                    raise AssertionError(f"Sibling import found in {path}: {node.module}")
