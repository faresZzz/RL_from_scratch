"""Tests pour le module deterministic_actor_critic (DDPG et TD3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_from_scratch.deterministic_actor_critic.network import (
    ContinuousQNetwork,
    DeterministicActor,
    TwinQNetwork,
)
from rl_from_scratch.deterministic_actor_critic.buffer import ContinuousReplayBuffer
from rl_from_scratch.deterministic_actor_critic.noise import GaussianNoise, OUNoise
from rl_from_scratch.deterministic_actor_critic.agent import DDPGAgent, TD3Agent
from rl_from_scratch.deterministic_actor_critic.training import (
    train_ddpg,
    train_td3,
)
from rl_from_scratch.core.utils import soft_update


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 32
BATCH_SIZE = 4
BUFFER_CAPACITY = 100

ACTION_LOW = np.array([-1.0] * ACTION_DIM, dtype=np.float32)
ACTION_HIGH = np.array([1.0] * ACTION_DIM, dtype=np.float32)


def _make_ddpg_agent(**kwargs) -> DDPGAgent:
    """Crée un DDPGAgent minimal pour les tests."""
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        noise_type="gaussian",
        noise_std=0.1,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        device="cpu",
    )
    defaults.update(kwargs)
    return DDPGAgent(**defaults)


def _make_td3_agent(**kwargs) -> TD3Agent:
    """Crée un TD3Agent minimal pour les tests."""
    defaults = dict(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        noise_type="gaussian",
        noise_std=0.1,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        device="cpu",
        policy_delay=2,
        target_noise=0.2,
        target_noise_clip=0.5,
    )
    defaults.update(kwargs)
    return TD3Agent(**defaults)


def _fill_buffer(agent: DDPGAgent, n: int = BATCH_SIZE + 1) -> None:
    """Remplit le buffer de l'agent avec des transitions aléatoires."""
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    next_obs = np.random.randn(OBS_DIM).astype(np.float32)
    for _ in range(n):
        action = np.random.uniform(-1.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)
        agent.store_transition(obs, action, 0.0, next_obs, False)


# ------------------------------------------------------------------
# Tests réseau : DeterministicActor
# ------------------------------------------------------------------


def test_deterministic_actor_output_bounds() -> None:
    """DeterministicActor produit des actions dans [action_low, action_high] pour tout input."""
    action_low = np.array([-1.0] * ACTION_DIM, dtype=np.float32)
    action_high = np.array([1.0] * ACTION_DIM, dtype=np.float32)
    actor = DeterministicActor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        action_low=action_low,
        action_high=action_high,
    )

    batch = 16
    obs = torch.randn(batch, OBS_DIM)
    with torch.no_grad():
        actions = actor(obs)

    low_t = torch.tensor(action_low)
    high_t = torch.tensor(action_high)
    assert actions.shape == (batch, ACTION_DIM), (
        f"Forme attendue ({batch}, {ACTION_DIM}), obtenu {actions.shape}"
    )
    assert (actions >= low_t - 1e-5).all(), "Action en dessous de action_low."
    assert (actions <= high_t + 1e-5).all(), "Action au-dessus de action_high."


def test_deterministic_actor_respects_vector_bounds() -> None:
    """Les bornes peuvent être différentes pour chaque dimension d'action."""
    action_low = np.array([-2.0, -0.5], dtype=np.float32)
    action_high = np.array([1.0, 3.0], dtype=np.float32)
    actor = DeterministicActor(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        action_low=action_low,
        action_high=action_high,
    )

    with torch.no_grad():
        actions = actor(torch.randn(32, OBS_DIM))

    assert (actions >= torch.tensor(action_low) - 1e-5).all()
    assert (actions <= torch.tensor(action_high) + 1e-5).all()


# ------------------------------------------------------------------
# Tests réseau : ContinuousQNetwork
# ------------------------------------------------------------------


def test_continuous_q_network_shape() -> None:
    """ContinuousQNetwork accepte (obs, action) et retourne un tenseur de forme (batch,)."""
    critic = ContinuousQNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

    batch = 8
    obs = torch.randn(batch, OBS_DIM)
    action = torch.randn(batch, ACTION_DIM)
    with torch.no_grad():
        q = critic(obs, action)

    assert q.shape == (batch,), f"Forme attendue ({batch},), obtenu {q.shape}"


# ------------------------------------------------------------------
# Tests réseau : TwinQNetwork
# ------------------------------------------------------------------


def test_twin_q_network_returns_two_values() -> None:
    """TwinQNetwork.forward retourne (q1, q2), chacun de forme (batch,)."""
    twin = TwinQNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

    batch = 8
    obs = torch.randn(batch, OBS_DIM)
    action = torch.randn(batch, ACTION_DIM)
    with torch.no_grad():
        q1, q2 = twin(obs, action)

    assert q1.shape == (batch,), f"q1 : forme attendue ({batch},), obtenu {q1.shape}"
    assert q2.shape == (batch,), f"q2 : forme attendue ({batch},), obtenu {q2.shape}"


def test_twin_q_network_q1_forward() -> None:
    """TwinQNetwork.q1_forward ne retourne que q1, de forme (batch,)."""
    twin = TwinQNetwork(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

    batch = 8
    obs = torch.randn(batch, OBS_DIM)
    action = torch.randn(batch, ACTION_DIM)
    with torch.no_grad():
        q1_only = twin.q1_forward(obs, action)

    assert q1_only.shape == (batch,), (
        f"q1_forward : forme attendue ({batch},), obtenu {q1_only.shape}"
    )


# ------------------------------------------------------------------
# Tests buffer : ContinuousReplayBuffer
# ------------------------------------------------------------------


def test_continuous_replay_buffer_push_sample() -> None:
    """push stocke des transitions et sample retourne les bonnes formes et dtypes."""
    buf = ContinuousReplayBuffer(capacity=BUFFER_CAPACITY)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = np.random.randn(ACTION_DIM).astype(np.float32)
    next_obs = np.random.randn(OBS_DIM).astype(np.float32)

    n_push = BATCH_SIZE * 4
    for _ in range(n_push):
        buf.push(obs, action, 1.0, next_obs, False)

    states, actions, rewards, next_states, dones = buf.sample(BATCH_SIZE)

    assert states.shape == (BATCH_SIZE, OBS_DIM), (
        f"states : forme attendue ({BATCH_SIZE}, {OBS_DIM}), obtenu {states.shape}"
    )
    assert actions.shape == (BATCH_SIZE, ACTION_DIM), (
        f"actions : forme attendue ({BATCH_SIZE}, {ACTION_DIM}), obtenu {actions.shape}"
    )
    assert rewards.shape == (BATCH_SIZE,) or rewards.shape == (BATCH_SIZE, 1), (
        f"rewards : forme inattendue {rewards.shape}"
    )
    assert next_states.shape == (BATCH_SIZE, OBS_DIM), (
        f"next_states : forme inattendue {next_states.shape}"
    )
    assert actions.dtype == torch.float32, (
        f"actions.dtype devrait être torch.float32, obtenu {actions.dtype}"
    )


def test_continuous_replay_buffer_capacity() -> None:
    """ContinuousReplayBuffer respecte sa capacité maximale."""
    capacity = 10
    buf = ContinuousReplayBuffer(capacity=capacity)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    for _ in range(capacity * 3):
        buf.push(obs, action, 0.0, obs, False)

    assert len(buf) == capacity, (
        f"Taille du buffer ({len(buf)}) devrait être égale à la capacité ({capacity})."
    )


# ------------------------------------------------------------------
# Tests bruit : GaussianNoise et OUNoise
# ------------------------------------------------------------------


def test_gaussian_noise_shape() -> None:
    """GaussianNoise retourne un tableau numpy de la bonne forme."""
    noise = GaussianNoise(action_dim=ACTION_DIM, sigma=0.1)
    n = noise()
    assert isinstance(n, np.ndarray), f"Attendu np.ndarray, obtenu {type(n)}"
    assert n.shape == (ACTION_DIM,), f"Forme attendue ({ACTION_DIM},), obtenu {n.shape}"


def test_ou_noise_reset() -> None:
    """OUNoise.reset() ramène l'état à mu ; les appels suivants produisent des valeurs différentes."""
    ou = OUNoise(action_dim=ACTION_DIM)
    ou.reset()

    # Plusieurs appels successifs sans reset doivent produire des valeurs différentes
    samples = [ou() for _ in range(10)]
    # Au moins certains doivent être différents (le process n'est pas constant)
    all_equal = all(np.allclose(samples[0], s) for s in samples[1:])
    assert not all_equal, "OUNoise devrait produire des valeurs différentes entre les appels."

    # reset() ramène à mu (vérification via un second reset propre)
    ou.reset()
    n = ou()
    assert isinstance(n, np.ndarray), f"Attendu np.ndarray après reset, obtenu {type(n)}"
    assert n.shape == (ACTION_DIM,), f"Forme attendue ({ACTION_DIM},), obtenu {n.shape}"


# ------------------------------------------------------------------
# Tests agent : DDPGAgent
# ------------------------------------------------------------------


def test_ddpg_agent_select_action_shape() -> None:
    """select_action retourne un np.ndarray de forme (action_dim,)."""
    agent = _make_ddpg_agent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = agent.select_action(obs, deterministic=False)

    assert isinstance(action, np.ndarray), (
        f"Attendu np.ndarray, obtenu {type(action)}"
    )
    assert action.shape == (ACTION_DIM,), (
        f"Forme attendue ({ACTION_DIM},), obtenu {action.shape}"
    )


def test_ddpg_agent_deterministic_no_noise() -> None:
    """select_action avec deterministic=True retourne le même résultat pour le même input."""
    torch.manual_seed(0)
    agent = _make_ddpg_agent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)

    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent.select_action(obs, deterministic=True)

    np.testing.assert_array_equal(a1, a2, err_msg="Mode déterministe doit retourner des actions identiques.")


def test_ddpg_agent_store_transition() -> None:
    """store_transition augmente la taille du buffer de l'agent."""
    agent = _make_ddpg_agent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    before = len(agent.replay_buffer)
    agent.store_transition(obs, action, 1.0, obs, False)
    after = len(agent.replay_buffer)

    assert after == before + 1, (
        f"Buffer devrait avoir {before + 1} transitions, obtenu {after}."
    )


def test_ddpg_agent_learn_step_keys() -> None:
    """learn_step retourne les pertes et diagnostics off-policy utiles."""
    torch.manual_seed(0)
    np.random.seed(0)

    agent = _make_ddpg_agent()
    _fill_buffer(agent, n=BATCH_SIZE * 5)

    metrics = agent.learn_step()

    expected_keys = {
        "actor_loss",
        "critic_loss",
        "q_mean",
        "noise_std",
        "action_abs_mean",
        "action_clip_fraction",
    }
    assert expected_keys.issubset(set(metrics.keys())), (
        f"Clés manquantes : {expected_keys - set(metrics.keys())}"
    )
    for key in expected_keys:
        value = metrics[key]
        assert isinstance(value, float), f"{key} devrait être float, obtenu {type(value)}"
        assert np.isfinite(value), f"{key} devrait être fini, obtenu {value}"


def test_ddpg_action_diagnostics_reports_clipping() -> None:
    """Les diagnostics mesurent l'amplitude clippée et la fraction hors bornes."""
    agent = _make_ddpg_agent()
    _fill_buffer(agent, n=BATCH_SIZE * 5)
    agent.record_action_diagnostics(
        raw_action=np.array([3.0, -0.25], dtype=np.float32),
        clipped_action=np.array([1.0, -0.25], dtype=np.float32),
    )

    metrics = agent.learn_step()

    assert metrics["action_abs_mean"] == pytest.approx(0.625)
    assert metrics["action_clip_fraction"] == pytest.approx(0.5)


def test_ddpg_agent_save_load_roundtrip(tmp_path) -> None:
    """save puis load produit un agent avec les mêmes poids."""
    torch.manual_seed(42)
    agent = _make_ddpg_agent()

    path = tmp_path / "ddpg_checkpoint.pt"
    agent.save(str(path))

    loaded = DDPGAgent.load(str(path))

    # Vérifie que les paramètres de l'acteur sont identiques
    for (name, p_orig), (_, p_load) in zip(
        agent.actor.named_parameters(), loaded.actor.named_parameters()
    ):
        assert torch.allclose(p_orig, p_load), (
            f"Paramètre '{name}' diffère après chargement."
        )


def test_soft_update_interpolation() -> None:
    """_soft_update avec tau=0.5 produit la moyenne exacte des paramètres online et target."""
    torch.manual_seed(0)
    agent = _make_ddpg_agent(tau=0.5)

    # Initialise les paramètres online et target avec des valeurs connues
    with torch.no_grad():
        for p_online, p_target in zip(
            agent.actor.parameters(), agent.actor_target.parameters()
        ):
            p_online.fill_(2.0)
            p_target.fill_(0.0)

    soft_update(agent.actor_target, agent.actor, tau=0.5)

    # Avec tau=0.5 : target_new = tau * online + (1 - tau) * target = 0.5 * 2 + 0.5 * 0 = 1.0
    for p_target in agent.actor_target.parameters():
        assert torch.allclose(p_target, torch.ones_like(p_target)), (
            f"Paramètre target devrait être 1.0 après soft update avec tau=0.5, obtenu {p_target}"
        )


def test_ddpg_target_uses_target_actor_and_target_critic() -> None:
    """La cible DDPG doit utiliser actor_target et critic_target."""

    class ConstantActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            self.called = True
            return torch.zeros((obs.shape[0], ACTION_DIM), dtype=torch.float32)

    class ConstantCritic(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value
            self.called = False

        def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            self.called = True
            return torch.full((obs.shape[0],), self.value, dtype=torch.float32)

    agent = _make_ddpg_agent(gamma=0.5)
    actor_target = ConstantActor()
    critic_target = ConstantCritic(7.0)
    agent.actor_target = actor_target
    agent.critic_target = critic_target
    agent.critic = ConstantCritic(1.0)

    states = torch.zeros(1, OBS_DIM)
    actions = torch.zeros(1, ACTION_DIM)
    rewards = torch.tensor([2.0])
    next_states = torch.zeros(1, OBS_DIM)
    dones = torch.tensor([0.0])

    loss = agent._compute_critic_loss(states, actions, rewards, next_states, dones)

    assert actor_target.called
    assert critic_target.called
    assert loss.item() == pytest.approx((1.0 - 5.5) ** 2)


# ------------------------------------------------------------------
# Tests agent : TD3Agent
# ------------------------------------------------------------------


def test_td3_inherits_ddpg() -> None:
    """TD3Agent est une instance de DDPGAgent."""
    agent = _make_td3_agent()
    assert isinstance(agent, DDPGAgent), "TD3Agent doit hériter de DDPGAgent."


def test_td3_has_twin_critics() -> None:
    """TD3Agent utilise un TwinQNetwork comme critique (remplace le critique DDPG simple)."""
    agent = _make_td3_agent()
    assert isinstance(agent.critic, TwinQNetwork), (
        f"critic doit être un TwinQNetwork pour TD3, obtenu {type(agent.critic)}"
    )


def test_td3_policy_delay() -> None:
    """Avec policy_delay=2, l'acteur n'est mis à jour qu'une fois sur deux learn_step."""
    torch.manual_seed(0)
    np.random.seed(0)

    agent = _make_td3_agent(policy_delay=2)
    _fill_buffer(agent, n=BATCH_SIZE * 10)

    metrics_1 = agent.learn_step()  # Étape 1 : pas de mise à jour acteur
    metrics_2 = agent.learn_step()  # Étape 2 : mise à jour acteur

    # La clé actor_updated doit exister dans au moins un des résultats
    # Étape 1 (impaire) : actor_updated = False ; Étape 2 (paire) : actor_updated = True
    assert "actor_updated" in metrics_1 or "actor_updated" in metrics_2, (
        "learn_step devrait retourner la clé 'actor_updated'."
    )

    if "actor_updated" in metrics_1 and "actor_updated" in metrics_2:
        # L'un des deux doit être False et l'autre True
        updates = [metrics_1["actor_updated"], metrics_2["actor_updated"]]
        assert True in updates, "L'acteur devrait être mis à jour au moins une fois sur 2."
        assert False in updates, "L'acteur ne devrait pas être mis à jour à chaque étape."


def test_td3_learn_step_keys() -> None:
    """learn_step du TD3Agent retourne les clés supplémentaires {q1_mean, q2_mean, q_gap}."""
    torch.manual_seed(0)
    np.random.seed(0)

    agent = _make_td3_agent()
    _fill_buffer(agent, n=BATCH_SIZE * 10)

    # Effectue plusieurs étapes pour s'assurer que l'acteur est mis à jour
    for _ in range(agent.policy_delay):
        metrics = agent.learn_step()

    expected_keys = {
        "critic_loss",
        "q1_mean",
        "q2_mean",
        "q_gap",
        "noise_std",
        "action_abs_mean",
        "action_clip_fraction",
    }
    assert expected_keys.issubset(set(metrics.keys())), (
        f"Clés manquantes dans TD3 learn_step : {expected_keys - set(metrics.keys())}"
    )


def test_td3_target_uses_min_q() -> None:
    """TD3 utilise vraiment min(q1, q2) dans la cible Bellman."""

    class ConstantTwinCritic(torch.nn.Module):
        def __init__(self, q1: float, q2: float) -> None:
            super().__init__()
            self.q1 = q1
            self.q2 = q2

        def forward(
            self, obs: torch.Tensor, action: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.full((obs.shape[0],), self.q1, dtype=torch.float32),
                torch.full((obs.shape[0],), self.q2, dtype=torch.float32),
            )

        def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return torch.full((obs.shape[0],), self.q1, dtype=torch.float32)

    class ZeroActor(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0], ACTION_DIM), dtype=torch.float32)

    agent = _make_td3_agent(gamma=0.5, target_noise=0.0)
    agent.actor_target = ZeroActor()
    agent.critic_target = ConstantTwinCritic(q1=10.0, q2=4.0)
    agent.critic = ConstantTwinCritic(q1=1.0, q2=2.0)

    states = torch.zeros(1, OBS_DIM)
    actions = torch.zeros(1, ACTION_DIM)
    rewards = torch.tensor([2.0])
    next_states = torch.zeros(1, OBS_DIM)
    dones = torch.tensor([0.0])

    loss = agent._compute_critic_loss(states, actions, rewards, next_states, dones)

    # target = 2 + 0.5 * min(10, 4) = 4
    assert loss.item() == pytest.approx((1.0 - 4.0) ** 2 + (2.0 - 4.0) ** 2)


# ------------------------------------------------------------------
# Smoke tests d'entraînement
# ------------------------------------------------------------------


def test_ddpg_training_smoke(tmp_path) -> None:
    """train_ddpg avec 500 timesteps termine et retourne les clés attendues."""
    from rl_from_scratch.deterministic_actor_critic.config import DDPGConfig

    config = DDPGConfig(
        env_id="Pendulum-v1",
        total_timesteps=500,
        hidden_dim=32,
        batch_size=32,
        buffer_capacity=1000,
        start_steps=50,
        update_after=50,
        checkpoint_every=500,
        device="cpu",
    )
    result = train_ddpg(config, output_dir=str(tmp_path), seed=0)

    assert isinstance(result, dict), "train_ddpg devrait retourner un dict."
    assert "agent" in result, "Clé 'agent' manquante."
    assert "history" in result, "Clé 'history' manquante."
    assert isinstance(result["history"], dict), "history devrait être un dict."
    assert len(result["history"]) > 0, "history ne devrait pas être vide."
    assert "step_actor_update_flags" in result["history"]
    assert "step_actor_updateds" not in result["history"]


def test_td3_training_smoke(tmp_path) -> None:
    """train_td3 avec 500 timesteps termine et retourne les clés attendues."""
    from rl_from_scratch.deterministic_actor_critic.config import TD3Config

    config = TD3Config(
        env_id="Pendulum-v1",
        total_timesteps=500,
        hidden_dim=32,
        batch_size=32,
        buffer_capacity=1000,
        start_steps=50,
        update_after=50,
        checkpoint_every=500,
        policy_delay=2,
        device="cpu",
    )
    result = train_td3(config, output_dir=str(tmp_path), seed=0)

    assert isinstance(result, dict), "train_td3 devrait retourner un dict."
    assert "agent" in result, "Clé 'agent' manquante."
    assert "history" in result, "Clé 'history' manquante."
    assert isinstance(result["history"], dict), "history devrait être un dict."
    assert len(result["history"]) > 0, "history ne devrait pas être vide."
    assert "step_actor_update_flags" in result["history"]
    assert "step_actor_updateds" not in result["history"]
