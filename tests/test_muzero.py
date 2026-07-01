"""Tests for the autonomous MuZero package."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import rl_from_scratch  # noqa: F401

from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY
from rl_from_scratch.muzero.agent import MuZeroAgent, _soft_cross_entropy
from rl_from_scratch.muzero.config import MuZeroConfig
from rl_from_scratch.muzero.connect_four import (
    ConnectFourAdapter,
    evaluate_vs_random,
    self_play_connect_four,
)
from rl_from_scratch.muzero.mcts import (
    MinMaxStats,
    Node,
    add_exploration_noise,
    backpropagate,
    run_mcts,
    ucb_score,
)
from rl_from_scratch.muzero.networks import (
    Dynamics,
    Prediction,
    Representation,
    inverse_scalar_transform,
    scalar_to_support,
    scalar_transform,
    scale_hidden_01,
    support_to_scalar,
)
from rl_from_scratch.muzero.replay import GameHistory, ReplayBuffer, make_target
from rl_from_scratch.muzero.training import train_muzero


@pytest.fixture(autouse=True)
def _float32():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


def _disable_muzero_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_from_scratch.core.reporting as _core_reporting
    import rl_from_scratch.muzero.reporting as _muzero_reporting

    monkeypatch.setattr(_muzero_reporting, "generate_training_figures", lambda *a, **kw: [])
    monkeypatch.setattr(_core_reporting, "record_greedy_episode", lambda *a, **kw: None)


def _make_config(**overrides: object) -> MuZeroConfig:
    base = dict(
        env_id="CartPole-v1",
        seed=0,
        total_timesteps=200,
        training_steps=8,
        max_steps_per_episode=20,
        num_simulations=4,
        num_unroll_steps=3,
        td_steps=3,
        batch_size=8,
        hidden_dim=16,
        encoding_dim=8,
        support_size=4,
        replay_capacity=32,
        num_warmup_games=2,
        selfplay_episodes_per_iteration=1,
        updates_per_iteration=2,
        eval_every=1,
        eval_episodes=1,
        checkpoint_every=1,
        discount=0.99,
        output_dir="runs-test",
        device="cpu",
    )
    base.update(overrides)
    return MuZeroConfig(**base)


def _make_agent(seed: int = 0, **overrides: object) -> MuZeroAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = int(overrides.pop("obs_dim", 4))
    num_actions = int(overrides.pop("num_actions", 2))
    config = _make_config(**overrides)
    return MuZeroAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        **config.to_dict(),
    )


def _make_game() -> GameHistory:
    game = GameHistory()
    game.observations = [
        np.array([0.0, 0.1], dtype=np.float32),
        np.array([1.0, 1.1], dtype=np.float32),
        np.array([2.0, 2.1], dtype=np.float32),
        np.array([3.0, 3.1], dtype=np.float32),
    ]
    game.actions = [1, 0, 1]
    game.rewards = [0.5, 1.5, -0.25]
    game.root_values = [0.7, 0.9, 0.2]
    game.child_visits = [
        np.array([0.2, 0.8], dtype=np.float32),
        np.array([0.6, 0.4], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]
    game.to_play = [1, 1, 1]
    return game


def _simple_model_fns():
    def prediction(hidden_state: torch.Tensor):
        batch = hidden_state.shape[0]
        priors = torch.tensor([[0.8, 0.2]], dtype=torch.float32).repeat(batch, 1)
        values = torch.full((batch,), 0.5, dtype=torch.float32)
        return priors.log(), values

    def dynamics(hidden_state: torch.Tensor, action: torch.Tensor):
        next_hidden = hidden_state + action.float().unsqueeze(-1) * 0.1 + 0.05
        rewards = 0.2 + action.float() * 0.05
        return next_hidden, rewards

    return {"prediction": prediction, "dynamics": dynamics}


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before: list[torch.Tensor], after_module: torch.nn.Module) -> bool:
    after = list(after_module.parameters())
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_scalar_transform_roundtrip() -> None:
    x = torch.tensor([-100.0, -5.0, -0.25, 0.0, 0.25, 5.0, 100.0], dtype=torch.float32)
    recovered = inverse_scalar_transform(scalar_transform(x))
    assert torch.allclose(recovered, x, atol=1e-3, rtol=1e-3)


def test_support_roundtrip_and_clipping() -> None:
    values = torch.tensor([-10.0, -3.7, 0.0, 4.2, 10.0], dtype=torch.float32)
    support = scalar_to_support(values, support_size=4)
    decoded = support_to_scalar(support, support_size=4)

    assert support.shape == (5, 9)
    assert torch.allclose(support.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.all(decoded <= 4.0 + 1e-6)
    assert torch.all(decoded >= -4.0 - 1e-6)
    assert decoded[1].item() == pytest.approx(-3.7, abs=0.25)
    assert decoded[3].item() == pytest.approx(4.0, abs=1e-4)


def test_network_shapes_and_latent_scaling() -> None:
    representation = Representation(obs_dim=4, hidden_dim=16, encoding_dim=8)
    dynamics = Dynamics(encoding_dim=8, action_dim=2, hidden_dim=16, support_size=4)
    prediction = Prediction(encoding_dim=8, hidden_dim=16, action_dim=2, support_size=4)

    obs = torch.randn(6, 4)
    hidden = representation(obs)
    scaled = scale_hidden_01(hidden)
    next_hidden, reward_logits = dynamics(scaled, torch.tensor([0, 1, 0, 1, 0, 1]))
    policy_logits, value_logits = prediction(next_hidden)

    assert hidden.shape == (6, 8)
    assert torch.all(scaled >= 0.0) and torch.all(scaled <= 1.0)
    assert next_hidden.shape == (6, 8)
    assert reward_logits.shape == (6, 9)
    assert policy_logits.shape == (6, 2)
    assert value_logits.shape == (6, 9)

    other_hidden, _ = dynamics(scaled[:1], torch.tensor([0]))
    changed_hidden, _ = dynamics(scaled[:1], torch.tensor([1]))
    assert not torch.allclose(other_hidden, changed_hidden)


def test_min_max_stats_is_finite_when_bounds_collapse() -> None:
    stats = MinMaxStats()
    stats.update(2.0)
    value = stats.normalize(2.0)
    assert math.isfinite(value)


def test_ucb_score_favors_higher_prior_for_unvisited_children() -> None:
    cfg = _make_config(num_simulations=2)
    min_max = MinMaxStats()
    parent = Node(prior=1.0, to_play=1)
    parent.visit_count = 10
    child_a = Node(prior=0.9, to_play=1)
    child_b = Node(prior=0.1, to_play=1)
    score_a = ucb_score(parent, child_a, min_max, cfg)
    score_b = ucb_score(parent, child_b, min_max, cfg)
    assert score_a > score_b


def test_run_mcts_respects_visit_budget_and_masks_illegal_actions() -> None:
    cfg = _make_config(num_simulations=8)
    root = Node(prior=1.0, to_play=1, hidden_state=torch.zeros(1, 8))
    run_mcts(
        root,
        _simple_model_fns(),
        cfg,
        legal_actions=np.array([1, 0], dtype=np.int8),
        to_play=1,
    )

    assert sum(child.visit_count for child in root.children.values()) == cfg.num_simulations
    assert set(root.children) == {0}


def test_run_mcts_uses_persistent_rng_for_root_noise() -> None:
    cfg = _make_config(num_simulations=1, dirichlet_alpha=0.3, exploration_fraction=0.8)
    rng = np.random.default_rng(123)

    root_a = Node(prior=1.0, to_play=1, hidden_state=torch.zeros(1, 8))
    root_b = Node(prior=1.0, to_play=1, hidden_state=torch.zeros(1, 8))
    run_mcts(root_a, _simple_model_fns(), cfg, rng=rng)
    run_mcts(root_b, _simple_model_fns(), cfg, rng=rng)

    priors_a = [root_a.children[action].prior for action in sorted(root_a.children)]
    priors_b = [root_b.children[action].prior for action in sorted(root_b.children)]
    assert priors_a != priors_b


def test_deterministic_mcts_does_not_add_root_noise() -> None:
    cfg = _make_config(num_simulations=1, exploration_fraction=0.8)
    root = Node(prior=1.0, to_play=1, hidden_state=torch.zeros(1, 8))

    run_mcts(root, _simple_model_fns(), cfg, add_root_noise=False)

    priors = [root.children[action].prior for action in sorted(root.children)]
    assert priors == pytest.approx([0.8, 0.2])


def test_dirichlet_noise_only_applies_at_root() -> None:
    cfg = _make_config(num_simulations=4, dirichlet_alpha=0.3, exploration_fraction=0.5)
    root = Node(prior=1.0, to_play=1, hidden_state=torch.zeros(1, 8))
    run_mcts(root, _simple_model_fns(), cfg, legal_actions=np.array([1, 1], dtype=np.int8), to_play=1)

    root_priors = [child.prior for child in root.children.values()]
    assert any(abs(prior - expected) > 1e-3 for prior, expected in zip(root_priors, [0.8, 0.2]))

    non_root_priors: list[float] = []
    for child in root.children.values():
        for grandchild in child.children.values():
            non_root_priors.append(grandchild.prior)
    assert non_root_priors
    assert set(round(x, 4) for x in non_root_priors) <= {0.2, 0.8}


def test_add_exploration_noise_changes_root_priors() -> None:
    root = Node(prior=1.0, to_play=1)
    root.children = {0: Node(prior=0.7, to_play=1), 1: Node(prior=0.3, to_play=1)}
    original = [child.prior for child in root.children.values()]
    add_exploration_noise(root, dirichlet_alpha=0.3, exploration_fraction=0.5, rng=np.random.default_rng(0))
    updated = [child.prior for child in root.children.values()]
    assert updated != original
    assert pytest.approx(sum(updated), abs=1e-6) == 1.0


def test_backpropagate_single_player_uses_reward_and_discount() -> None:
    stats = MinMaxStats()
    root = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=1, reward=2.0)
    search_path = [root, child]
    backpropagate(search_path, value=3.0, discount=0.5, min_max_stats=stats, two_player=False)

    assert child.value() == pytest.approx(3.0)
    assert root.value() == pytest.approx(2.0 + 0.5 * 3.0)


def test_backpropagate_two_player_flips_sign() -> None:
    stats = MinMaxStats()
    root = Node(prior=1.0, to_play=1)
    child = Node(prior=0.5, to_play=-1, reward=1.0)
    search_path = [root, child]
    backpropagate(search_path, value=2.0, discount=1.0, min_max_stats=stats, two_player=True)

    assert child.value() == pytest.approx(2.0)
    assert root.value() == pytest.approx(-1.0)


def test_two_player_selection_prefers_move_good_for_parent() -> None:
    # Negamax convention: a child is the opponent's node, so a LOW child value is
    # GREAT for the parent. Selection must flip the child's sign and pick it.
    from types import SimpleNamespace

    from rl_from_scratch.muzero.mcts import select_child

    config = SimpleNamespace(pb_c_base=19_652.0, pb_c_init=1.25, discount=1.0, two_player=True)
    parent = Node(prior=1.0, to_play=1, visit_count=2)
    good = Node(prior=0.5, to_play=-1, visit_count=1, value_sum=-0.9)  # opponent value low
    bad = Node(prior=0.5, to_play=-1, visit_count=1, value_sum=0.9)    # opponent value high
    parent.children = {0: good, 1: bad}
    stats = MinMaxStats()
    stats.update(-0.9)
    stats.update(0.9)
    action, _ = select_child(parent, stats, config)
    assert action == 0  # the move that is good for the parent


def test_make_target_alignment_and_terminal_bootstrap_disabled() -> None:
    game = _make_game()
    cfg = _make_config(num_unroll_steps=3, td_steps=2, discount=0.5)
    targets = make_target(game, 0, cfg)

    assert len(targets) == 4
    first = targets[0]
    terminal = targets[-1]

    assert np.allclose(first["observation"], game.observations[0])
    assert first["action"] == game.actions[0]
    assert first["reward"] == pytest.approx(game.rewards[0])
    assert first["value"] == pytest.approx(0.5 + 0.5 * 1.5 + 0.25 * game.root_values[2])
    assert np.allclose(first["policy"], game.child_visits[0])

    assert terminal["action"] == 0
    assert terminal["reward"] == 0.0
    assert terminal["value"] == 0.0
    assert first["policy_mask"] == pytest.approx(1.0)
    assert terminal["policy_mask"] == pytest.approx(0.0)


def test_replay_buffer_sampling_uses_local_seed() -> None:
    games = [_make_game(), _make_game(), _make_game()]
    buffer_a = ReplayBuffer(capacity=8, seed=7)
    buffer_b = ReplayBuffer(capacity=8, seed=7)
    buffer_c = ReplayBuffer(capacity=8, seed=9)
    for game in games:
        buffer_a.add_game(game)
        buffer_b.add_game(game)
        buffer_c.add_game(game)

    sample_a = buffer_a.sample_positions(4)
    sample_b = buffer_b.sample_positions(4)
    sample_c = buffer_c.sample_positions(4)

    assert sample_a == sample_b
    assert sample_a != sample_c


def test_replay_buffer_samples_positions_not_games_uniformly() -> None:
    short = _make_game()
    long = _make_game()
    long.observations = [np.asarray([float(i), 0.0], dtype=np.float32) for i in range(21)]
    long.actions = [0] * 20
    long.rewards = [0.0] * 20
    long.root_values = [0.0] * 20
    long.child_visits = [np.asarray([0.5, 0.5], dtype=np.float32) for _ in range(20)]
    long.to_play = [1] * 20

    buffer = ReplayBuffer(capacity=4, seed=123)
    buffer.add_game(short)
    buffer.add_game(long)

    samples = buffer.sample_positions(1_000)
    long_fraction = sum(1 for game_index, _ in samples if game_index == 1) / len(samples)

    # Long game has 20 positions, short game has 3 positions: P(long) ≈ 20/23.
    assert long_fraction > 0.75


def test_muzero_policy_loss_ignores_fictitious_post_terminal_targets() -> None:
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    masks = torch.tensor([1.0, 0.0])

    masked = _soft_cross_entropy(logits, targets, weights=masks)
    first_only = _soft_cross_entropy(logits[:1], targets[:1])

    assert masked == pytest.approx(first_only)


def test_muzero_config_registered() -> None:
    assert "muzero" in CONFIG_REGISTRY
    assert CONFIG_REGISTRY["muzero"] is MuZeroConfig


def test_muzero_agent_factory_registered() -> None:
    assert "muzero" in AGENT_FACTORIES
    assert AGENT_FACTORIES["muzero"] is train_muzero


def test_muzero_config_validation_and_roundtrip() -> None:
    with pytest.raises(ValueError, match="support_size"):
        MuZeroConfig(support_size=0)
    with pytest.raises(ValueError, match="num_unroll_steps"):
        MuZeroConfig(num_unroll_steps=0)
    with pytest.raises(ValueError, match="discount"):
        MuZeroConfig(discount=1.5)

    config = _make_config(hidden_dim=32, discount=0.997)
    restored = MuZeroConfig.from_dict(config.to_dict())
    assert restored.hidden_dim == 32
    assert restored.discount == pytest.approx(0.997)


def test_muzero_select_action_is_legal() -> None:
    agent = _make_agent()
    obs = np.zeros(4, dtype=np.float32)
    action = agent.select_action(obs, legal_actions=np.array([0, 1], dtype=np.int8))
    assert action == 1


def test_muzero_evaluate_restores_acting_state() -> None:
    from rl_from_scratch.muzero.training import evaluate

    agent = _make_agent(num_simulations=2)
    obs = np.zeros(4, dtype=np.float32)
    agent.select_action(obs)
    before = agent.snapshot_acting_state()

    evaluate(agent, "CartPole-v1", n_episodes=1, seed=123, max_steps=3)
    after = agent.snapshot_acting_state()

    assert after["pending_root_value"] == before["pending_root_value"]
    assert np.allclose(after["pending_child_visits"], before["pending_child_visits"])
    assert after["pending_to_play"] == before["pending_to_play"]
    assert after["episode_index"] == before["episode_index"]


def test_muzero_learn_step_is_finite_and_updates_params() -> None:
    agent = _make_agent(batch_size=2, num_unroll_steps=2, td_steps=2, replay_capacity=8)
    game = _make_game()
    game.observations = [
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        np.full(4, 2.0, dtype=np.float32),
        np.full(4, 3.0, dtype=np.float32),
    ]
    agent.replay_buffer.add_game(game)
    agent.replay_buffer.add_game(game)

    before = _clone_parameters(agent.prediction)
    metrics = agent.learn_step()

    expected = {"loss", "policy_loss", "value_loss", "reward_loss", "root_value_mean"}
    assert expected.issubset(metrics)
    for value in metrics.values():
        assert math.isfinite(float(value))
    assert _parameters_changed(before, agent.prediction)


def test_muzero_save_load_roundtrip(tmp_path: Path) -> None:
    agent = _make_agent()
    checkpoint = agent.save(tmp_path / "muzero.pt")
    loaded = MuZeroAgent.load(checkpoint, device="cpu")
    obs = np.zeros(4, dtype=np.float32)
    original = agent.select_action(obs, deterministic=True)
    restored = loaded.select_action(obs, deterministic=True)
    assert original == restored


def test_train_muzero_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_muzero_figures(monkeypatch)
    config = _make_config(output_dir=str(tmp_path))
    result = train_muzero(config, seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    assert isinstance(result["agent"], MuZeroAgent)
    assert isinstance(result["history"], dict)
    assert result["paths"].run_dir.exists()


def test_connect_four_adapter_and_self_play() -> None:
    pytest.importorskip("pettingzoo")
    adapter = ConnectFourAdapter()
    obs, legal_actions = adapter.reset(seed=0)
    assert obs.shape == (84,)
    assert legal_actions.shape == (7,)
    assert legal_actions.sum() > 0

    agent = _make_agent(
        obs_dim=84,
        num_actions=7,
        hidden_dim=16,
        encoding_dim=16,
        support_size=4,
        two_player=True,
        discount=1.0,
    )
    game = self_play_connect_four(agent, max_moves=6, seed=0)
    assert len(game.observations) >= 1

    score = evaluate_vs_random(agent, num_games=1, max_moves=6, seed=0)
    assert set(score) >= {"mean_reward", "win_rate"}


def test_connect_four_opponent_reward_is_scored_from_player_zero_perspective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_from_scratch.muzero.connect_four as cf

    class _AlwaysOpponentAgent:
        def select_action(self, *args, **kwargs):
            return 0

    class _FakeAdapter:
        def __init__(self):
            self.calls = 0

        def reset(self, seed=None):
            return np.zeros(84, dtype=np.float32), np.ones(7, dtype=np.int8)

        def step(self, action: int):
            self.calls += 1
            if self.calls == 1:
                return (
                    np.zeros(84, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {},
                    np.ones(7, dtype=np.int8),
                    -1,
                )
            return (
                np.zeros(84, dtype=np.float32),
                1.0,  # reward received by the acting opponent
                True,
                False,
                {},
                np.ones(7, dtype=np.int8),
                -1,
            )

        def close(self):
            pass

    monkeypatch.setattr(cf, "ConnectFourAdapter", _FakeAdapter)
    result = cf.evaluate_vs_random(_AlwaysOpponentAgent(), num_games=1, max_moves=2)
    assert result["mean_reward"] == pytest.approx(-1.0)
    assert result["win_rate"] == pytest.approx(0.0)


def test_muzero_has_no_cross_package_imports() -> None:
    muzero_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "rl_from_scratch"
        / "muzero"
    )
    forbidden = {
        "rl_from_scratch.actor_critic",
        "rl_from_scratch.deep_q",
        "rl_from_scratch.deterministic_actor_critic",
        "rl_from_scratch.dreamer",
        "rl_from_scratch.dyna",
        "rl_from_scratch.mbpo",
        "rl_from_scratch.pets",
        "rl_from_scratch.pilco",
        "rl_from_scratch.reinforce",
        "rl_from_scratch.sac",
        "rl_from_scratch.tabular",
        "rl_from_scratch.trust_region",
    }

    for path in sorted(muzero_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for pkg in forbidden:
                        assert not alias.name.startswith(pkg), (
                            f"Cross-package import of '{alias.name}' found in {path.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for pkg in forbidden:
                    assert not module.startswith(pkg), (
                        f"Cross-package import 'from {module}' found in {path.name}"
                    )
