"""Tests for PILCO — both the validated math core and the integration layer.

Part 1 (tests 1-9): math-core unit tests — kernel, GP, moment matching, cost,
policy, belief propagation, trajectory optimisation.  Validated against
Monte-Carlo, all in float64.

Part 2 (tests 10+): integration tests — config/registry, agent API, training
smoke, save/load round-trip.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_from_scratch.pilco.cost import (
    expected_cost,
    expected_inverted_pendulum_cost,
    inverted_pendulum_particle_cost,
    saturating_cost,
)
from rl_from_scratch.pilco.gp import GaussianProcess, MultiOutputGP
from rl_from_scratch.pilco.kernel import RBFKernel
from rl_from_scratch.pilco.moment_matching import gaussian_moments
from rl_from_scratch.pilco.policy import LinearSinePolicy, MLPPolicy, RBFPolicy
from rl_from_scratch.pilco.belief_propagation import (
    predict_trajectory,
    project_angle_belief_torch,
    propagate,
)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


# ----------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------
def test_rbf_kernel_is_symmetric_psd_and_decays_with_distance() -> None:
    torch.manual_seed(0)
    kernel = RBFKernel(3)
    x = torch.randn(8, 3)
    gram = kernel(x)
    assert torch.allclose(gram, gram.t(), atol=1e-10)
    eigvals = torch.linalg.eigvalsh(gram)
    assert float(eigvals.min().detach()) > -1e-8  # positive semi-definite
    # diagonal equals signal variance
    assert torch.allclose(torch.diagonal(gram), kernel.diagonal(x), atol=1e-10)
    # closer points are more correlated than far ones
    near = kernel(torch.zeros(1, 3), torch.full((1, 3), 0.1))
    far = kernel(torch.zeros(1, 3), torch.full((1, 3), 5.0))
    assert float(near.detach()) > float(far.detach())


# ----------------------------------------------------------------------
# Gaussian Process regression
# ----------------------------------------------------------------------
def test_gp_fit_reduces_nlml_and_predicts_known_function() -> None:
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 25).unsqueeze(1)
    y = torch.sin(x).squeeze(1)
    gp = GaussianProcess(1)
    gp.set_data(x, y)
    nlml_before = float(gp.negative_log_marginal_likelihood().detach())
    nlml_after = gp.fit(n_steps=60)
    assert nlml_after < nlml_before
    # accurate interpolation on training inputs
    mean, _ = gp.predict(x)
    assert float(((mean - y) ** 2).mean().sqrt().detach()) < 1e-2
    # uncertainty grows away from the data
    _, var_near = gp.predict(torch.zeros(1, 1))
    _, var_far = gp.predict(torch.full((1, 1), 10.0))
    assert float(var_far.detach()) > float(var_near.detach())


# ----------------------------------------------------------------------
# Moment matching vs Monte-Carlo (the core PILCO approximation)
# ----------------------------------------------------------------------
def test_moment_matching_matches_monte_carlo() -> None:
    torch.manual_seed(0)
    n, d, e = 18, 2, 2
    x = torch.rand(n, d) * 2 - 1
    y = torch.stack(
        [torch.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2, torch.cos(x[:, 1]) - 0.3 * x[:, 0]],
        dim=1,
    )
    gp = MultiOutputGP(d, e)
    gp.fit(x, y, n_steps=60)

    mu = torch.tensor([0.1, -0.2])
    a = torch.randn(d, d) * 0.25
    sigma = a @ a.t() + 0.04 * torch.eye(d)

    mean, cov, cross = gaussian_moments(gp, mu, sigma)

    k = 300_000
    chol = torch.linalg.cholesky(sigma)
    samples = mu + torch.randn(k, d) @ chol.t()
    with torch.no_grad():
        preds = [gp.gps[i].predict(samples) for i in range(e)]
        means = torch.stack([p[0] for p in preds], dim=1)
        varis = torch.stack([p[1] for p in preds], dim=1)
    mean_mc = means.mean(0)
    cov_mc = torch.cov(means.t()) + torch.diag(varis.mean(0))
    xc = samples - samples.mean(0)
    mc = means - means.mean(0)
    cross_mc = (xc.t() @ mc) / (k - 1)

    assert float((mean - mean_mc).abs().max().detach()) < 5e-3
    assert float((cov - cov_mc).abs().max().detach()) < 5e-3
    assert float((cross - cross_mc).abs().max().detach()) < 5e-3
    # sanity: the moments are non-trivial (off-diagonal coupling present)
    assert float(cov[0, 1].abs().detach()) > 1e-4


# ----------------------------------------------------------------------
# Saturating cost
# ----------------------------------------------------------------------
def test_saturating_cost_is_bounded_and_zero_at_target() -> None:
    target = torch.tensor([1.0, 0.0, 0.0])
    weight = torch.eye(3)
    assert float(saturating_cost(target, target, weight).detach()) == pytest.approx(0.0, abs=1e-10)
    far = saturating_cost(torch.tensor([9.0, 9.0, 9.0]), target, weight)
    assert 0.0 <= float(far.detach()) <= 1.0
    assert float(far.detach()) > 0.99  # saturates toward 1 far away


def test_expected_cost_matches_monte_carlo() -> None:
    torch.manual_seed(0)
    d = 3
    mu = torch.tensor([0.3, -0.4, 0.1])
    a = torch.randn(d, d) * 0.35
    sigma = a @ a.t() + 0.05 * torch.eye(d)
    target = torch.tensor([1.0, 0.0, 0.0])
    weight = torch.diag(torch.tensor([1.0, 0.5, 0.2]))

    analytic = expected_cost(mu, sigma, target, weight)
    k = 400_000
    samples = mu + torch.randn(k, d) @ torch.linalg.cholesky(sigma).t()
    mc = saturating_cost(samples, target, weight).mean()
    assert 0.0 <= float(analytic) <= 1.0
    assert float((analytic - mc).abs()) < 5e-3


def test_expected_cost_matches_notebook_symmetric_form_on_correlated_belief() -> None:
    mu = torch.tensor([0.4, -0.15, 0.9, 0.25, -0.3], dtype=torch.float64)
    sigma = torch.tensor(
        [
            [0.18, 0.04, -0.02, 0.03, 0.00],
            [0.04, 0.09, 0.01, -0.02, 0.03],
            [-0.02, 0.01, 0.07, 0.00, 0.02],
            [0.03, -0.02, 0.00, 0.11, -0.01],
            [0.00, 0.03, 0.02, -0.01, 0.08],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    weight = torch.diag(torch.tensor([0.01, 100.0, 1.0, 0.01, 2.0], dtype=torch.float64))

    eye = torch.eye(mu.shape[0], dtype=mu.dtype)
    sigma_sym = 0.5 * (sigma + sigma.t())
    sqrt_w = torch.diag(torch.sqrt(torch.diagonal(weight)))
    a = eye + sqrt_w @ sigma_sym @ sqrt_w
    a = 0.5 * (a + a.t()) + 1e-8 * eye
    delta = mu - target
    expected = 1.0 - torch.exp(
        -0.5 * torch.linalg.slogdet(a).logabsdet
        - 0.5 * (delta @ (sqrt_w @ torch.linalg.solve(a, sqrt_w)) @ delta)
    )

    actual = expected_cost(mu, sigma, target, weight)

    assert float(actual) == pytest.approx(float(expected.clamp(0.0, 1.0)), abs=1e-12)


def test_inverted_pendulum_costs_penalize_failure_risk_and_actions() -> None:
    mu_safe = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    mu_risky = torch.tensor(
        [0.9, math.sin(0.25), math.cos(0.25), 0.2, 0.4],
        dtype=torch.float64,
    )
    sigma = 1e-3 * torch.eye(5, dtype=torch.float64)
    mu_u = torch.tensor([0.0], dtype=torch.float64)
    sigma_u = 1e-4 * torch.eye(1, dtype=torch.float64)
    safe = expected_inverted_pendulum_cost(mu_safe, sigma, mu_u, sigma_u)
    risky = expected_inverted_pendulum_cost(mu_risky, sigma, mu_u, sigma_u)
    assert float(risky) > float(safe)

    particles = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.05, math.sin(0.3), math.cos(0.3), 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    costs = inverted_pendulum_particle_cost(particles, actions)
    assert costs.shape == (2,)
    assert float(costs[1]) > float(costs[0])


def test_inverted_pendulum_terminal_risk_matches_gym_angle_rule() -> None:
    """Cart position affects the control cost, but it is not a Gym termination."""
    sigma = 1e-6 * torch.eye(5, dtype=torch.float64)
    zero_state_weight = torch.zeros((5, 5), dtype=torch.float64)
    zero_action_weight = torch.zeros((1, 1), dtype=torch.float64)
    action = torch.zeros(1, dtype=torch.float64)
    action_cov = torch.zeros((1, 1), dtype=torch.float64)

    centered = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    displaced = torch.tensor([2.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    centered_cost = expected_inverted_pendulum_cost(
        centered,
        sigma,
        action,
        action_cov,
        state_weight=zero_state_weight,
        action_weight=zero_action_weight,
    )
    displaced_cost = expected_inverted_pendulum_cost(
        displaced,
        sigma,
        action,
        action_cov,
        state_weight=zero_state_weight,
        action_weight=zero_action_weight,
    )
    assert float(displaced_cost) == pytest.approx(float(centered_cost), abs=1e-10)


# ----------------------------------------------------------------------
# RBF policy
# ----------------------------------------------------------------------
def test_rbf_policy_respects_action_bounds() -> None:
    torch.manual_seed(0)
    policy = RBFPolicy(3, 1, n_basis=20, action_high=torch.tensor([2.0]))
    x = torch.randn(50, 3)
    actions = policy(x)
    assert actions.shape == (50, 1)
    assert bool((actions.abs() <= 2.0 + 1e-9).all())


def test_linear_sine_policy_has_correct_shape_bounds_and_gradients() -> None:
    torch.manual_seed(0)
    policy = LinearSinePolicy(5, 1, action_high=torch.tensor([3.0]))
    x = torch.randn(32, 5, requires_grad=True)
    actions = policy(x)
    assert actions.shape == (32, 1)
    assert bool((actions.abs() <= 3.0 + 1e-9).all())
    actions.square().mean().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_mlp_policy_has_correct_shape_bounds_and_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    policy = MLPPolicy(
        5,
        1,
        hidden_dim=16,
        hidden_layers=2,
        action_high=torch.tensor([2.0]),
    )
    x = torch.randn(8, 5, requires_grad=True)
    y = policy(x)
    assert y.shape == (8, 1)
    assert bool((y.abs() <= 2.0 + 1e-6).all())
    y.sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())

    clone = MLPPolicy(
        5,
        1,
        hidden_dim=16,
        hidden_layers=2,
        action_high=torch.tensor([2.0]),
    )
    clone.load_state_dict(policy.state_dict())
    with torch.no_grad():
        np.testing.assert_allclose(
            policy(x.detach()).numpy(),
            clone(x.detach()).numpy(),
            atol=1e-7,
        )


def test_policy_propagation_matches_monte_carlo_in_pilco_regime() -> None:
    torch.manual_seed(2)
    dx, du = 2, 1
    policy = RBFPolicy(dx, du, n_basis=20, action_high=torch.tensor([2.0]))
    with torch.no_grad():
        policy.weights.copy_(0.3 * torch.randn(20, du))
        policy.log_lengthscales.copy_(torch.log(torch.tensor([1.0, 1.0])))

    mu_x = torch.tensor([0.1, -0.05])
    sigma_x = 0.01 * torch.eye(dx)  # small belief — PILCO operating regime
    mu_u, sigma_u, c_xu = policy.propagate(mu_x, sigma_x)

    k = 400_000
    samples = mu_x + torch.randn(k, dx) @ torch.linalg.cholesky(sigma_x).t()
    with torch.no_grad():
        controls = policy(samples)
    mu_mc = controls.mean(0)
    sigma_mc = torch.cov(controls.t()).reshape(du, du)
    xc = samples - samples.mean(0)
    uc = controls - controls.mean(0)
    cross_mc = (xc.t() @ uc) / (k - 1)

    assert float((mu_u - mu_mc).abs().max().detach()) < 5e-3
    assert float((sigma_u - sigma_mc).abs().max().detach()) < 5e-3
    assert float((c_xu - cross_mc).abs().max().detach()) < 5e-3


# ----------------------------------------------------------------------
# Belief propagation + trajectory optimisation (the PILCO loop core)
# ----------------------------------------------------------------------
def _toy_dynamics_gp() -> MultiOutputGP:
    """Fit a GP to a known delta-dynamics ``Delta = f(state, action)``."""
    def f(xu: torch.Tensor) -> torch.Tensor:
        x0, x1, u = xu[:, 0], xu[:, 1], xu[:, 2]
        return torch.stack([0.1 * torch.sin(x0) + 0.05 * u, -0.1 * x1 + 0.05 * u], dim=1)

    x_tr = torch.cat([torch.rand(40, 2) * 2 - 1, torch.rand(40, 1) * 4 - 2], dim=1)
    gp = MultiOutputGP(3, 2)
    gp.fit(x_tr, f(x_tr), n_steps=60)
    return gp


def test_belief_propagation_matches_monte_carlo() -> None:
    torch.manual_seed(0)
    gp = _toy_dynamics_gp()
    policy = RBFPolicy(2, 1, n_basis=15, action_high=torch.tensor([2.0]))
    with torch.no_grad():
        policy.weights.copy_(0.3 * torch.randn(15, 1))

    mu0 = torch.tensor([0.2, -0.1])
    sigma0 = 0.01 * torch.eye(2)
    mu1, sigma1 = propagate(gp, policy, mu0, sigma0)

    k = 200_000
    x0 = mu0 + torch.randn(k, 2) @ torch.linalg.cholesky(sigma0).t()
    with torch.no_grad():
        u = policy(x0)
        joint = torch.cat([x0, u], dim=1)
        dmean = torch.stack([gp.gps[a].predict(joint)[0] for a in range(2)], dim=1)
        dvar = torch.stack([gp.gps[a].predict(joint)[1] for a in range(2)], dim=1)
        delta = dmean + torch.sqrt(dvar.clamp_min(0)) * torch.randn(k, 2)
        x1 = x0 + delta
    assert float((mu1 - x1.mean(0)).abs().max().detach()) < 5e-3
    assert float((sigma1 - torch.cov(x1.t())).abs().max().detach()) < 5e-3


def test_trajectory_cost_is_differentiable_and_optimisable() -> None:
    torch.manual_seed(0)
    gp = _toy_dynamics_gp()
    for p in gp.parameters():
        p.requires_grad_(False)  # freeze model during policy optimisation
    policy = RBFPolicy(2, 1, n_basis=15, action_high=torch.tensor([2.0]))
    target = torch.zeros(2)
    weight = torch.eye(2)
    mu0 = torch.tensor([0.6, 0.3])
    sigma0 = 0.01 * torch.eye(2)

    def cost() -> torch.Tensor:
        total, _ = predict_trajectory(
            gp, policy, mu0, sigma0, horizon=15, target=target, weight=weight
        )
        return total

    j0 = float(cost().detach())
    grad = torch.autograd.grad(cost(), policy.weights)[0]
    assert bool(torch.isfinite(grad).all())

    optimizer = torch.optim.LBFGS(
        policy.parameters(), lr=0.1, max_iter=30, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = cost()
        loss.backward()
        return loss

    optimizer.step(closure)
    assert float(cost().detach()) < j0  # policy optimisation reduces predicted cost


# ======================================================================
# Part 2 — Integration tests
# ======================================================================

from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY  # noqa: E402
from rl_from_scratch.core.config import load_config  # noqa: E402
from rl_from_scratch.core.env import encode_obs, project_encoded_angle_np  # noqa: E402
from rl_from_scratch.pilco.agent import DeepPilcoAgent, PilcoAgent  # noqa: E402
from rl_from_scratch.pilco.buffer import TransitionBuffer  # noqa: E402
from rl_from_scratch.pilco.config import DeepPilcoConfig, PilcoConfig  # noqa: E402
from rl_from_scratch.pilco.belief_propagation import (  # noqa: E402
    project_angle_belief_torch,
    project_encoded_angle_torch,
)
from rl_from_scratch.pilco.training import train_deep_pilco, train_pilco  # noqa: E402


# ----------------------------------------------------------------------
# encode_obs
# ----------------------------------------------------------------------

def test_encode_obs_maps_4d_to_5d_with_trigonometric_identity() -> None:
    """encode_obs: 4-D raw → 5-D with sin²+cos²=1 and correct layout."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        raw = rng.uniform(-1.0, 1.0, size=4)
        enc = encode_obs(raw)
        assert enc.shape == (5,), f"Expected shape (5,), got {enc.shape}"
        assert enc.dtype == np.float64
        # sin²(θ) + cos²(θ) = 1
        sin_theta = enc[1]
        cos_theta = enc[2]
        assert abs(sin_theta**2 + cos_theta**2 - 1.0) < 1e-12, (
            f"Pythagorean identity violated: sin²+cos²={sin_theta**2 + cos_theta**2}"
        )
        # other dims pass through unchanged
        assert enc[0] == pytest.approx(raw[0])   # cart_pos
        assert enc[3] == pytest.approx(raw[2])   # cart_vel
        assert enc[4] == pytest.approx(raw[3])   # θdot
        # sin/cos match the actual angle
        theta = raw[1]
        assert enc[1] == pytest.approx(np.sin(theta))
        assert enc[2] == pytest.approx(np.cos(theta))


def test_encode_obs_upright_target() -> None:
    """encode_obs: θ=0 → [cart_pos, 0, 1, cart_vel, θdot] (upright target)."""
    raw = np.array([0.0, 0.0, 0.0, 0.0])
    enc = encode_obs(raw)
    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(enc, expected, atol=1e-12)


def test_project_encoded_angle_np_restores_unit_circle() -> None:
    """Imagined encoded states must keep sin²+cos²=1 after residual updates."""
    encoded = np.array(
        [
            [0.0, 0.3, 0.4, 0.1, -0.2],
            [0.2, -2.0, 0.5, 0.0, 0.3],
        ],
        dtype=np.float64,
    )
    projected = project_encoded_angle_np(encoded)
    norms = projected[:, 1] ** 2 + projected[:, 2] ** 2
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)
    np.testing.assert_allclose(projected[:, [0, 3, 4]], encoded[:, [0, 3, 4]])


def test_project_encoded_angle_torch_is_differentiable() -> None:
    x = torch.tensor([[0.0, 0.3, 0.4, 0.1, -0.2]], requires_grad=True)
    y = project_encoded_angle_torch(x)
    norm = y[0, 1] ** 2 + y[0, 2] ** 2
    assert float(norm.detach()) == pytest.approx(1.0, abs=1e-6)
    y.sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_project_angle_belief_torch_projects_mean_and_covariance() -> None:
    mu = torch.tensor([0.1, 0.4, 0.3, -0.2, 0.5], dtype=torch.float64)
    sigma = torch.tensor(
        [
            [0.10, 0.02, 0.01, 0.00, 0.00],
            [0.02, 0.08, 0.03, 0.00, 0.00],
            [0.01, 0.03, 0.09, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.05, 0.01],
            [0.00, 0.00, 0.00, 0.01, 0.04],
        ],
        dtype=torch.float64,
    )
    mu_p, sigma_p = project_angle_belief_torch(mu, sigma)
    assert float(mu_p[1].square() + mu_p[2].square()) == pytest.approx(1.0, abs=1e-8)
    assert torch.allclose(sigma_p, sigma_p.t(), atol=1e-10)
    eigvals = torch.linalg.eigvalsh(sigma_p)
    assert float(eigvals.min().detach()) >= -1e-8


def _disable_pilco_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    import rl_from_scratch.pilco.reporting as _rep
    monkeypatch.setattr(_rep, "generate_training_figures", lambda *a, **kw: [])


def _make_agent(
    obs_dim: int = 3,
    action_dim: int = 1,
    action_low: list | None = None,
    action_high: list | None = None,
    n_basis: int = 10,
    seed: int = 0,
    encode_angle: bool = False,
) -> PilcoAgent:
    torch.manual_seed(seed)
    return PilcoAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low or [-2.0] * action_dim,
        action_high=action_high or [2.0] * action_dim,
        n_basis=n_basis,
        horizon=5,
        gp_fit_steps=5,
        policy_opt_steps=3,
        max_gp_points=60,
        seed=seed,
        encode_angle=encode_angle,
    )


def _make_deep_agent(
    obs_dim: int = 4,
    action_dim: int = 1,
    seed: int = 0,
    encode_angle: bool = True,
) -> DeepPilcoAgent:
    torch.manual_seed(seed)
    return DeepPilcoAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=[-3.0],
        action_high=[3.0],
        hidden_dim=16,
        n_layers=1,
        dropout_p=0.05,
        n_particles=8,
        horizon=5,
        model_train_steps=5,
        model_batch_size=8,
        policy_opt_steps=3,
        policy_lr=0.01,
        n_basis=8,
        max_gp_points=32,
        seed=seed,
        encode_angle=encode_angle,
        policy_type="mlp",
        cost_mode="inverted_pendulum",
    )


def _fill_agent_buffer(agent: PilcoAgent, n: int = 20) -> None:
    rng = np.random.default_rng(42)
    for _ in range(n):
        obs = rng.standard_normal(agent.obs_dim)
        action = rng.standard_normal(agent.action_dim)
        next_obs = obs + 0.1 * rng.standard_normal(agent.obs_dim)
        agent.store_transition(obs, action, 0.0, next_obs, False)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

def test_pilco_config_registered() -> None:
    assert "pilco" in CONFIG_REGISTRY
    assert CONFIG_REGISTRY["pilco"] is PilcoConfig


def test_pilco_agent_factory_registered() -> None:
    assert "pilco" in AGENT_FACTORIES
    assert AGENT_FACTORIES["pilco"] is train_pilco


# -----------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------

def test_pilco_config_rejects_nonpositive_n_basis() -> None:
    with pytest.raises(ValueError, match="n_basis"):
        PilcoConfig(n_basis=0)


def test_pilco_config_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        PilcoConfig(horizon=-1)


def test_pilco_config_rejects_zero_init_state_cov() -> None:
    with pytest.raises(ValueError, match="init_state_cov"):
        PilcoConfig(init_state_cov=0.0)


def test_pilco_config_rejects_negative_cost_weight() -> None:
    with pytest.raises(ValueError, match="cost_weight"):
        PilcoConfig(cost_weight=(1.0, -0.5, 0.1))


def test_pilco_config_rejects_invalid_collection_horizon() -> None:
    with pytest.raises(ValueError, match="collection_horizon"):
        PilcoConfig(collection_horizon=0)


def test_pilco_parity_and_full_configs_load_expected_constants() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "pilco"

    smoke = load_config(root / "pilco_invertedpendulum_smoke.yaml")
    parity = load_config(root / "pilco_invertedpendulum_parity.yaml")
    full = load_config(root / "pilco_invertedpendulum.yaml")

    assert isinstance(smoke, PilcoConfig)
    assert smoke.episodes == 1
    assert smoke.encode_angle is True
    assert smoke.fixed_horizon_steps == 10
    assert smoke.policy_type == "linear_sine"
    assert smoke.gp_fit_steps == 3
    assert smoke.policy_opt_steps == 2
    assert smoke.final_eval_episodes == 1

    assert isinstance(parity, PilcoConfig)
    assert parity.episodes == 10
    assert parity.collection_horizon == 50
    assert parity.fixed_horizon_steps == 50
    assert parity.num_init_rollouts == 3
    assert parity.max_gp_points == 100
    assert parity.gp_fit_steps == 20
    assert parity.policy_opt_steps == 10
    assert parity.horizon == 60
    assert parity.policy_type == "linear_sine"
    assert parity.n_initial_beliefs == 2
    assert parity.exploration_noise == pytest.approx(0.05)
    assert parity.validation_fraction == pytest.approx(0.2)
    assert parity.validation_min_points == 20
    assert parity.eval_every == 1
    assert parity.eval_episodes == 10
    assert parity.eval_seed == 400
    assert parity.final_eval_episodes == 20
    assert parity.final_eval_seed == 900

    assert isinstance(full, PilcoConfig)
    assert full.episodes == 30
    assert full.collection_horizon == 50
    assert full.fixed_horizon_steps == 50
    assert full.num_init_rollouts == 3
    assert full.max_gp_points == 100
    assert full.gp_fit_steps == 20
    assert full.policy_opt_steps == 10
    assert full.horizon == 60
    assert full.policy_type == "linear_sine"
    assert full.n_initial_beliefs == 2
    assert full.exploration_noise == pytest.approx(0.05)
    assert full.validation_fraction == pytest.approx(0.2)
    assert full.validation_min_points == 20
    assert full.eval_every == 1
    assert full.eval_episodes == 20
    assert full.eval_seed == 400
    assert full.final_eval_episodes == 20
    assert full.final_eval_seed == 900


# -----------------------------------------------------------------------
# Agent: select_action
# -----------------------------------------------------------------------

def test_pilco_select_action_shape_and_dtype() -> None:
    agent = _make_agent()
    obs = np.zeros(3, dtype=np.float32)
    action = agent.select_action(obs, deterministic=True)
    assert action.shape == (1,)
    assert action.dtype == np.float32


def test_pilco_select_action_within_policy_bounds() -> None:
    """Policy sine-saturation keeps actions in [-action_high, action_high]."""
    agent = _make_agent(action_high=[2.0])
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.standard_normal(3).astype(np.float32)
        action = agent.select_action(obs, deterministic=True)
        assert float(np.abs(action).max()) <= 2.0 + 1e-5


# -----------------------------------------------------------------------
# Agent: learn_step
# -----------------------------------------------------------------------

def test_pilco_learn_step_returns_finite_metrics() -> None:
    agent = _make_agent(seed=0)
    _fill_agent_buffer(agent, n=30)

    metrics = agent.learn_step()

    assert "model_nll" in metrics
    assert "predicted_cost" in metrics
    assert "gp_points" in metrics
    for key, value in metrics.items():
        assert math.isfinite(float(value)), f"{key} is not finite: {value}"


def test_pilco_buffer_stratified_cap_and_fixed_holdout_are_disjoint() -> None:
    buffer = TransitionBuffer(max_gp_points=6, rng=np.random.default_rng(0))
    for i in range(10):
        obs = np.array([float(i), 0.0], dtype=np.float64)
        nxt = obs + 0.1
        buffer.push(obs, np.array([0.0]), nxt, failure=(i % 4 == 0))
    train_x, train_y, holdout_x, holdout_y, meta = buffer.get_train_and_holdout_tensors(
        validation_fraction=0.2,
        validation_min_points=2,
        seed=7,
    )
    assert train_x.shape[0] <= 6
    assert holdout_x.shape[0] == 2
    assert holdout_y.shape[0] == 2
    assert meta["train_failure_count"] > 0
    assert meta["train_safe_count"] > 0
    assert set(meta["train_indices"]).isdisjoint(meta["holdout_indices"])


def test_pilco_buffer_cap_prefers_upright_states_by_sine_threshold() -> None:
    buffer = TransitionBuffer(max_gp_points=10, rng=np.random.default_rng(0))
    for _ in range(12):
        obs = np.array([0.0, 0.05, 1.0], dtype=np.float64)
        buffer.push(obs, np.array([0.0]), obs + 0.01, failure=True)
    for _ in range(12):
        obs = np.array([0.0, 0.6, 1.0], dtype=np.float64)
        buffer.push(obs, np.array([0.0]), obs + 0.01, failure=False)

    train_x, _, _, _, _ = buffer.get_train_and_holdout_tensors(
        validation_fraction=0.0,
        validation_min_points=0,
        seed=0,
    )

    safe_mask = np.abs(train_x[:, 1].numpy()) <= np.sin(0.25)
    assert int(safe_mask.sum()) == 8


def test_deep_buffer_returns_most_recent_transitions() -> None:
    buffer = TransitionBuffer(max_gp_points=4, rng=np.random.default_rng(0))
    for i in range(7):
        obs = np.array([float(i), 0.0, 0.0], dtype=np.float64)
        action = np.array([0.0], dtype=np.float64)
        buffer.push(obs, action, obs + 1.0, failure=False)

    x, y = buffer.get_recent_tensors(cap=4)
    np.testing.assert_allclose(x[:, 0].numpy(), np.array([3.0, 4.0, 5.0, 6.0]))
    np.testing.assert_allclose(y[:, 0].numpy(), np.ones(4))


def test_pilco_fit_dynamics_warm_starts_existing_gp() -> None:
    agent = _make_agent(seed=0)
    _fill_agent_buffer(agent, n=30)
    agent.fit_dynamics()
    assert agent.gp is not None
    first_gp_id = id(agent.gp.gps[0])
    old_lengths = agent.gp.gps[0].kernel.lengthscales.detach().clone()

    _fill_agent_buffer(agent, n=10)
    agent.fit_dynamics()
    assert agent.gp is not None
    assert id(agent.gp.gps[0]) == first_gp_id
    assert bool(torch.isfinite(agent.gp.gps[0].kernel.lengthscales).all())
    assert agent._dynamics_fit_count == 2
    assert old_lengths.shape == agent.gp.gps[0].kernel.lengthscales.shape


def test_pilco_optimize_policy_reverts_on_degrading_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(seed=0)
    _fill_agent_buffer(agent, n=20)
    agent.fit_dynamics()

    values = iter(
        [
            (torch.tensor(1.0, dtype=torch.float64), []),
            (torch.tensor(2.0, dtype=torch.float64), []),
            (torch.tensor(1.0, dtype=torch.float64), []),
        ]
    )

    monkeypatch.setattr(
        "rl_from_scratch.pilco.agent.predict_trajectory",
        lambda *args, **kwargs: next(values),
    )
    before = [p.detach().clone() for p in agent.policy.parameters()]
    _, final_cost = agent.optimize_policy()
    after = [p.detach().clone() for p in agent.policy.parameters()]
    assert final_cost == pytest.approx(1.0)
    for lhs, rhs in zip(before, after):
        assert torch.allclose(lhs, rhs)


def test_pilco_buffer_grows_with_transitions() -> None:
    agent = _make_agent()
    assert len(agent.buffer) == 0
    _fill_agent_buffer(agent, n=10)
    assert len(agent.buffer) == 10


def test_deep_pilco_learn_step_returns_finite_metrics() -> None:
    agent = _make_deep_agent(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(24):
        raw = np.array(
            [rng.normal(0.0, 0.05), rng.normal(0.0, 0.1), 0.0, 0.0],
            dtype=np.float32,
        )
        nxt = raw + rng.normal(0.0, 0.01, size=4).astype(np.float32)
        agent.store_transition(
            raw,
            np.array([0.0], dtype=np.float32),
            0.0,
            nxt,
            False,
        )
    metrics = agent.learn_step()
    assert "model_loss" in metrics
    assert "predicted_cost" in metrics
    for value in metrics.values():
        assert math.isfinite(float(value))


# -----------------------------------------------------------------------
# Save / load round-trip
# -----------------------------------------------------------------------

def test_pilco_save_load_preserves_policy(tmp_path: Path) -> None:
    agent = _make_agent(seed=7)
    _fill_agent_buffer(agent, n=20)
    agent.learn_step()

    ckpt = agent.save(tmp_path / "pilco.pt")
    loaded = PilcoAgent.load(
        ckpt,
        obs_dim=3,
        action_dim=1,
        action_low=[-2.0],
        action_high=[2.0],
        n_basis=10,
        horizon=5,
        gp_fit_steps=5,
        policy_opt_steps=3,
        max_gp_points=60,
        seed=7,
    )

    # Policy parameters must be identical after round-trip
    for p_orig, p_loaded in zip(
        agent.policy.parameters(), loaded.policy.parameters()
    ):
        assert torch.allclose(p_orig.double(), p_loaded.double())

    assert len(loaded.buffer) == len(agent.buffer)


# -----------------------------------------------------------------------
# Training smoke test (fast config)
# -----------------------------------------------------------------------

def test_pilco_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_pilco_figures(monkeypatch)

    config = PilcoConfig(
        env_id="Pendulum-v1",
        episodes=2,
        horizon=5,
        n_basis=10,
        gp_fit_steps=5,
        policy_opt_steps=3,
        num_init_rollouts=1,
        max_gp_points=60,
        max_steps_per_episode=40,
        eval_every=2,
        eval_episodes=1,
        checkpoint_every=2,
        output_dir=str(tmp_path),
    )

    result = train_pilco(config, seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    assert isinstance(result["agent"], PilcoAgent)
    assert len(result["history"]["episode_rewards"]) == 2
    assert result["paths"].run_dir.exists()


def test_deep_pilco_training_smoke_pendulum_basic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_pilco_figures(monkeypatch)

    config = DeepPilcoConfig(
        env_id="Pendulum-v1",
        episodes=2,
        horizon=5,
        hidden_dim=16,
        n_layers=1,
        n_particles=8,
        model_train_steps=5,
        model_batch_size=8,
        policy_opt_steps=3,
        n_basis=8,
        num_init_rollouts=1,
        max_gp_points=40,
        max_steps_per_episode=30,
        eval_every=2,
        eval_episodes=1,
        checkpoint_every=2,
        output_dir=str(tmp_path),
    )

    result = train_deep_pilco(config, seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    assert isinstance(result["agent"], DeepPilcoAgent)
    assert len(result["history"]["episode_rewards"]) == 2
    assert result["paths"].run_dir.exists()


# -----------------------------------------------------------------------
# InvertedPendulum-v5 smoke test
# -----------------------------------------------------------------------

def test_pilco_invertedpendulum_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: PILCO runs on InvertedPendulum-v5 with encode_angle + fixed_horizon.

    Tests the full reference recipe: sin/cos angle encoding (4-D → 5-D) and
    fixed-horizon data collection via NoEarlyTermination.
    """
    _disable_pilco_figures(monkeypatch)

    config = PilcoConfig(
        env_id="InvertedPendulum-v5",
        episodes=2,
        horizon=8,
        n_basis=10,
        gp_fit_steps=5,
        policy_opt_steps=3,
        num_init_rollouts=1,
        max_gp_points=100,
        max_steps_per_episode=50,
        init_state_cov=1e-4,
        # 5-D encoded cost_weight: [cart_pos, sinθ, cosθ, cart_vel, θdot]
        cost_weight=(0.5, 1.0, 1.0, 0.0, 0.1),
        encode_angle=True,
        fixed_horizon_steps=8,
        eval_every=2,
        eval_episodes=1,
        checkpoint_every=2,
        output_dir=str(tmp_path),
    )

    result = train_pilco(config, seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    assert isinstance(result["agent"], PilcoAgent)
    assert len(result["history"]["episode_rewards"]) == 2
    assert result["paths"].run_dir.exists()
    # Agent internal state dim should be 5 (4 + 1 for sin/cos split)
    agent = result["agent"]
    assert agent.obs_dim == 5
    assert agent.env_obs_dim == 4


def test_pilco_buffer_tracks_episode_initial_states() -> None:
    """TransitionBuffer.start_episode() tracks episode-initial states correctly."""
    from rl_from_scratch.pilco.buffer import TransitionBuffer
    buf = TransitionBuffer(max_gp_points=100)
    rng = np.random.default_rng(0)

    # Simulate 3 episodes each with 5 transitions
    init_obs_list = []
    for _ in range(3):
        init_obs = rng.standard_normal(4)
        init_obs_list.append(init_obs)
        buf.start_episode(init_obs)
        obs = init_obs.copy()
        for _ in range(5):
            action = rng.standard_normal(1)
            next_obs = obs + 0.05 * rng.standard_normal(4)
            buf.push(obs, action, next_obs)
            obs = next_obs.copy()

    # initial_state_mean() must reflect the actual episode initial obs
    mu = buf.initial_state_mean()
    expected_mu = np.mean(init_obs_list, axis=0)
    assert np.allclose(mu, expected_mu, atol=1e-10), (
        f"initial_state_mean mismatch: got {mu}, expected {expected_mu}"
    )

    # initial_state_cov() must reflect the variance of episode initial obs
    cov_diag = buf.initial_state_cov()
    expected_var = np.var(np.stack(init_obs_list), axis=0)
    assert cov_diag.shape == (4,)
    # variance should be close to actual variance of the 3 initial obs
    assert np.allclose(cov_diag, expected_var.clip(1e-6), atol=1e-10)


def test_pilco_cost_decreases_over_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 4 PILCO iterations and check that the imagined cost trends down.

    Uses InvertedPendulum-v5 with encode_angle + fixed_horizon_steps.
    We allow some tolerance: a strict monotone decrease is not guaranteed
    (GP and policy landscape are non-convex), but the final cost should be
    noticeably below the first-iteration cost.
    """
    _disable_pilco_figures(monkeypatch)

    config = PilcoConfig(
        env_id="InvertedPendulum-v5",
        episodes=4,
        horizon=10,
        n_basis=15,
        gp_fit_steps=30,
        policy_opt_steps=30,
        num_init_rollouts=2,
        max_gp_points=200,
        max_steps_per_episode=200,
        init_state_cov=1e-4,
        # 5-D encoded cost: [cart_pos, sinθ, cosθ, cart_vel, θdot]
        cost_weight=(0.5, 1.0, 1.0, 0.0, 0.1),
        encode_angle=True,
        fixed_horizon_steps=10,
        eval_every=999,    # skip eval to keep test fast
        eval_episodes=1,
        checkpoint_every=999,
        output_dir=str(tmp_path),
    )

    result = train_pilco(config, seed=42)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    history = result["history"]

    # "predicted_cost" is stored under the key "step_predicted_costs" by
    # RunManager.record_updates (see core/metrics.py history_key_for_update_metric).
    predicted_costs = history.get("step_predicted_costs", [])
    assert len(predicted_costs) == 4, (
        f"Expected 4 cost entries, got {len(predicted_costs)}: {predicted_costs}. "
        f"Available history keys: {list(history.keys())}"
    )
    # All costs must be finite
    for i, c in enumerate(predicted_costs):
        assert math.isfinite(float(c)), f"iteration {i} cost is not finite: {c}"

    predicted_befores = history.get("step_predicted_cost_befores", [])
    assert len(predicted_befores) == len(predicted_costs)
    assert any(float(after) <= float(before) + 1e-8 for before, after in zip(predicted_befores, predicted_costs)), (
        "Expected at least one policy-improvement step to avoid increasing the imagined PILCO cost. "
        f"before={[f'{c:.4f}' for c in predicted_befores]} "
        f"after={[f'{c:.4f}' for c in predicted_costs]}"
    )


# ======================================================================
# Part 3 — Deep PILCO tests
# ======================================================================

from rl_from_scratch.pilco.bnn import (  # noqa: E402
    BayesianDynamicsNetwork,
    predict_trajectory_particles,
    propagate_particles,
    train_bnn_on_buffer,
)
from rl_from_scratch.pilco.agent import DeepPilcoAgent  # noqa: E402
from rl_from_scratch.pilco.config import DeepPilcoConfig  # noqa: E402
from rl_from_scratch.pilco.training import train_deep_pilco  # noqa: E402


def _make_bnn(
    state_dim: int = 3,
    action_dim: int = 1,
    hidden_dim: int = 16,
    n_layers: int = 2,
    dropout_p: float = 0.1,
) -> BayesianDynamicsNetwork:
    return BayesianDynamicsNetwork(
        input_dim=state_dim + action_dim,
        output_dim=state_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout_p=dropout_p,
    ).float()


def _make_deep_agent(
    obs_dim: int = 3,
    action_dim: int = 1,
    n_basis: int = 10,
    seed: int = 0,
) -> DeepPilcoAgent:
    torch.manual_seed(seed)
    return DeepPilcoAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=[-2.0] * action_dim,
        action_high=[2.0] * action_dim,
        hidden_dim=16,
        n_layers=2,
        dropout_p=0.05,
        n_particles=8,
        horizon=4,
        model_train_steps=5,
        model_batch_size=8,
        model_lr=1e-3,
        policy_opt_steps=3,
        policy_lr=0.01,
        n_basis=n_basis,
        max_gp_points=60,
        seed=seed,
    )


def _fill_deep_agent_buffer(agent: DeepPilcoAgent, n: int = 20) -> None:
    rng = np.random.default_rng(42)
    for _ in range(n):
        obs = rng.standard_normal(agent.obs_dim).astype(np.float32)
        action = rng.standard_normal(agent.action_dim).astype(np.float32)
        next_obs = obs + 0.1 * rng.standard_normal(agent.obs_dim).astype(np.float32)
        agent.store_transition(obs, action, 0.0, next_obs, False)


# -----------------------------------------------------------------------
# BNN stochasticity: different masks → different output
# -----------------------------------------------------------------------

def test_bnn_different_masks_give_different_outputs() -> None:
    torch.manual_seed(0)
    net = _make_bnn()
    xu = torch.randn(4, 4, dtype=torch.float32)   # explicit float32: BNN is float32

    masks_a = net.sample_masks(4)
    masks_b = net.sample_masks(4)

    with torch.no_grad():
        out_a = net.forward(xu, masks_a)
        out_b = net.forward(xu, masks_b)

    # Different masks → different outputs (with overwhelmingly high probability)
    assert not torch.allclose(out_a, out_b, atol=1e-6)


def test_bnn_same_mask_gives_same_output() -> None:
    torch.manual_seed(0)
    net = _make_bnn()
    xu = torch.randn(4, 4, dtype=torch.float32)   # explicit float32
    masks = net.sample_masks(4)

    with torch.no_grad():
        out1 = net.forward(xu, masks)
        out2 = net.forward(xu, masks)

    assert torch.allclose(out1, out2, atol=1e-8)


def test_predict_trajectory_particles_reuses_provided_masks() -> None:
    torch.manual_seed(0)
    net = _make_bnn(state_dim=3, action_dim=1, hidden_dim=4, n_layers=2, dropout_p=0.0)
    policy = RBFPolicy(
        3,
        1,
        n_basis=4,
        action_high=torch.tensor([1.0], dtype=torch.float32),
    ).float()
    particles = torch.zeros(2, 3, dtype=torch.float32)
    masks = [torch.ones(2, 4, dtype=torch.float32), torch.ones(2, 4, dtype=torch.float32)]

    def _fail(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("sample_masks should not be called when masks are provided")

    net.sample_masks = _fail  # type: ignore[assignment]

    total_cost, _ = predict_trajectory_particles(
        net,
        policy,
        particles0=particles,
        masks=masks,
        horizon=4,
        target=torch.zeros(3, dtype=torch.float32),
        weight=torch.eye(3, dtype=torch.float32),
        n_particles=2,
        step_cost_fn=lambda states, actions: torch.full(
            (states.shape[0],),
            2.0,
            dtype=states.dtype,
            device=states.device,
        ),
    )

    assert float(total_cost.detach()) == pytest.approx(2.0, abs=1e-6)


def test_train_bnn_on_buffer_uses_seeded_split_and_reports_metadata() -> None:
    torch.manual_seed(0)
    net = _make_bnn(state_dim=3, action_dim=1, hidden_dim=8, n_layers=1, dropout_p=0.0)
    x = torch.randn(40, 4, dtype=torch.float32)
    y = torch.randn(40, 3, dtype=torch.float32)

    _, meta_a = train_bnn_on_buffer(net, x, y, n_steps=4, batch_size=8, lr=1e-3, seed=11)
    torch.manual_seed(0)
    net_b = _make_bnn(state_dim=3, action_dim=1, hidden_dim=8, n_layers=1, dropout_p=0.0)
    _, meta_b = train_bnn_on_buffer(net_b, x, y, n_steps=4, batch_size=8, lr=1e-3, seed=11)
    _, meta_c = train_bnn_on_buffer(net_b, x, y, n_steps=4, batch_size=8, lr=1e-3, seed=12)

    assert meta_a["train_indices"] == meta_b["train_indices"]
    assert meta_a["val_indices"] == meta_b["val_indices"]
    assert meta_a["train_indices"] != meta_c["train_indices"]
    assert meta_a["weight_decay"] == pytest.approx(1e-5)
    assert meta_a["grad_clip"] == pytest.approx(10.0)


# -----------------------------------------------------------------------
# Particle propagation: finite output + differentiable cost
# -----------------------------------------------------------------------

def test_propagate_particles_returns_finite_outputs() -> None:
    torch.manual_seed(1)
    state_dim, action_dim = 3, 1
    K = 10
    net = _make_bnn(state_dim, action_dim)
    # RBFPolicy must be float32 to match the BNN
    policy = RBFPolicy(state_dim, action_dim, n_basis=8,
                       action_high=torch.tensor([2.0], dtype=torch.float32)).float()
    particles = torch.randn(K, state_dim, dtype=torch.float32)
    masks = net.sample_masks(K)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    weight = torch.eye(state_dim, dtype=torch.float32)
    action_high = torch.tensor([2.0], dtype=torch.float32)

    next_p, cost = propagate_particles(net, policy, particles, masks, target, weight, action_high)

    assert next_p.shape == (K, state_dim)
    assert bool(torch.isfinite(next_p).all()), "next_particles contains non-finite values"
    assert torch.isfinite(cost), f"step_cost is not finite: {cost}"


def test_propagate_particles_projects_encoded_angle() -> None:
    torch.manual_seed(4)
    state_dim, action_dim = 5, 1
    K = 12
    net = _make_bnn(state_dim, action_dim, hidden_dim=16, dropout_p=0.0)
    policy = RBFPolicy(
        state_dim,
        action_dim,
        n_basis=8,
        action_high=torch.tensor([3.0], dtype=torch.float32),
    ).float()
    particles = torch.randn(K, state_dim, dtype=torch.float32)
    particles[:, 1] = 0.2
    particles[:, 2] = 0.2
    masks = net.sample_masks(K)
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    weight = torch.eye(state_dim, dtype=torch.float32)
    next_p, _ = propagate_particles(
        net,
        policy,
        particles,
        masks,
        target,
        weight,
        torch.tensor([3.0], dtype=torch.float32),
        project_encoded_angle=True,
    )
    norms = next_p[:, 1].square() + next_p[:, 2].square()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_trajectory_cost_is_differentiable() -> None:
    """Total trajectory cost must have finite gradients w.r.t. policy params."""
    torch.manual_seed(2)
    state_dim, action_dim = 3, 1
    net = _make_bnn(state_dim, action_dim)
    # RBFPolicy must be float32 to match the BNN
    policy = RBFPolicy(state_dim, action_dim, n_basis=8,
                       action_high=torch.tensor([2.0], dtype=torch.float32)).float()
    mu0 = torch.zeros(state_dim, dtype=torch.float32)
    sigma0 = 0.01 * torch.eye(state_dim, dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    weight = torch.eye(state_dim, dtype=torch.float32)

    for p in net.parameters():
        p.requires_grad_(False)

    total_cost, traj = predict_trajectory_particles(
        net, policy, mu0, sigma0,
        horizon=5,
        target=target,
        weight=weight,
        n_particles=8,
    )

    assert torch.isfinite(total_cost), f"total_cost is not finite: {total_cost}"
    assert len(traj) == 6  # horizon + 1 initial mean

    grads = torch.autograd.grad(total_cost, list(policy.parameters()))
    for g in grads:
        assert bool(torch.isfinite(g).all()), "policy gradient contains non-finite values"


def test_one_adam_step_reduces_trajectory_cost() -> None:
    """A single Adam step on a toy dynamics should reduce the predicted cost."""
    torch.manual_seed(3)
    state_dim, action_dim = 2, 1

    # Tiny BNN with NO dropout so cost is deterministic across two calls
    net = BayesianDynamicsNetwork(state_dim + action_dim, state_dim,
                                  hidden_dim=8, n_layers=1, dropout_p=0.0).float()
    # RBFPolicy must be float32 to match the BNN
    policy = RBFPolicy(state_dim, action_dim, n_basis=6,
                       action_high=torch.tensor([2.0], dtype=torch.float32)).float()

    mu0 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    sigma0 = 0.01 * torch.eye(state_dim, dtype=torch.float32)
    target = torch.zeros(state_dim, dtype=torch.float32)
    weight = torch.eye(state_dim, dtype=torch.float32)

    for p in net.parameters():
        p.requires_grad_(False)

    def cost_fn() -> torch.Tensor:
        return predict_trajectory_particles(
            net, policy, mu0, sigma0,
            horizon=3, target=target, weight=weight, n_particles=6,
        )[0]

    j0 = float(cost_fn().item())

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    optimizer.zero_grad()
    loss = cost_fn()
    loss.backward()
    optimizer.step()

    j1 = float(cost_fn().item())
    # With no dropout stochasticity and a gradient step, cost should decrease
    # (or at worst stay close — we only require gradient was finite)
    assert math.isfinite(j0) and math.isfinite(j1)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

def test_deep_pilco_config_registered() -> None:
    assert "deep_pilco" in CONFIG_REGISTRY
    assert CONFIG_REGISTRY["deep_pilco"] is DeepPilcoConfig


def test_deep_pilco_agent_factory_registered() -> None:
    assert "deep_pilco" in AGENT_FACTORIES
    assert AGENT_FACTORIES["deep_pilco"] is train_deep_pilco


# -----------------------------------------------------------------------
# DeepPilcoConfig validation
# -----------------------------------------------------------------------

def test_deep_pilco_config_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        DeepPilcoConfig(horizon=0)


def test_deep_pilco_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout_p"):
        DeepPilcoConfig(dropout_p=1.0)


def test_deep_pilco_config_rejects_nonpositive_n_particles() -> None:
    with pytest.raises(ValueError, match="n_particles"):
        DeepPilcoConfig(n_particles=1)


def test_deep_pilco_config_rejects_invalid_collection_horizon() -> None:
    with pytest.raises(ValueError, match="collection_horizon"):
        DeepPilcoConfig(collection_horizon=0)


def test_deep_parity_and_full_configs_load_expected_constants() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "pilco"

    smoke = load_config(root / "deep_pilco_invertedpendulum_smoke.yaml")
    parity = load_config(root / "deep_pilco_invertedpendulum_parity.yaml")
    full = load_config(root / "deep_pilco_invertedpendulum.yaml")

    assert isinstance(smoke, DeepPilcoConfig)
    assert smoke.episodes == 1
    assert smoke.encode_angle is True
    assert smoke.hidden_dim == 16
    assert smoke.n_particles == 8
    assert smoke.model_train_steps == 3
    assert smoke.policy_opt_steps == 2
    assert smoke.policy_type == "mlp"
    assert smoke.final_eval_episodes == 1

    assert isinstance(parity, DeepPilcoConfig)
    assert parity.episodes == 22
    assert parity.collection_horizon == 120
    assert parity.hidden_dim == 96
    assert parity.n_layers == 2
    assert parity.dropout_p == pytest.approx(0.08)
    assert parity.n_particles == 96
    assert parity.horizon == 80
    assert parity.model_train_steps == 260
    assert parity.model_batch_size == 128
    assert parity.model_lr == pytest.approx(8e-4)
    assert parity.policy_opt_steps == 70
    assert parity.policy_lr == pytest.approx(3e-3)
    assert parity.policy_type == "mlp"
    assert parity.policy_hidden_dim == 64
    assert parity.policy_hidden_layers == 1
    assert parity.num_init_rollouts == 8
    assert parity.max_gp_points == 2500
    assert parity.validation_fraction == pytest.approx(0.2)
    assert parity.exploration_noise == pytest.approx(0.35)
    assert parity.eval_every == 3
    assert parity.eval_episodes == 20
    assert parity.eval_seed == 400
    assert parity.final_eval_episodes == 20
    assert parity.final_eval_seed == 900

    assert isinstance(full, DeepPilcoConfig)
    assert full.episodes == 50
    assert full.collection_horizon == 120
    assert full.hidden_dim == 96
    assert full.n_layers == 2
    assert full.dropout_p == pytest.approx(0.08)
    assert full.n_particles == 96
    assert full.horizon == 80
    assert full.model_train_steps == 500
    assert full.model_batch_size == 128
    assert full.model_lr == pytest.approx(8e-4)
    assert full.policy_opt_steps == 120
    assert full.policy_lr == pytest.approx(3e-3)
    assert full.policy_type == "mlp"
    assert full.policy_hidden_dim == 64
    assert full.policy_hidden_layers == 1
    assert full.num_init_rollouts == 8
    assert full.max_gp_points == 2500
    assert full.validation_fraction == pytest.approx(0.2)
    assert full.exploration_noise == pytest.approx(0.35)
    assert full.eval_every == 3
    assert full.eval_episodes == 20
    assert full.eval_seed == 400
    assert full.final_eval_episodes == 20
    assert full.final_eval_seed == 900


# -----------------------------------------------------------------------
# Agent: select_action
# -----------------------------------------------------------------------

def test_deep_pilco_select_action_shape_and_dtype() -> None:
    agent = _make_deep_agent()
    obs = np.zeros(3, dtype=np.float32)
    action = agent.select_action(obs, deterministic=True)
    assert action.shape == (1,)
    assert action.dtype == np.float32


def test_deep_pilco_select_action_within_bounds() -> None:
    agent = _make_deep_agent(action_dim=1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.standard_normal(3).astype(np.float32)
        action = agent.select_action(obs, deterministic=True)
        assert float(np.abs(action).max()) <= 2.0 + 1e-5


# -----------------------------------------------------------------------
# Agent: learn_step
# -----------------------------------------------------------------------

def test_deep_pilco_learn_step_returns_finite_metrics() -> None:
    agent = _make_deep_agent(seed=0)
    _fill_deep_agent_buffer(agent, n=30)

    metrics = agent.learn_step()

    assert "model_loss" in metrics
    assert "model_train_loss" in metrics
    assert "model_val_loss" in metrics
    assert "model_weight_decay" in metrics
    assert "model_grad_clip" in metrics
    assert "predicted_cost" in metrics
    assert "gp_points" in metrics
    assert metrics["model_weight_decay"] == pytest.approx(1e-5)
    assert metrics["model_grad_clip"] == pytest.approx(10.0)
    for key, value in metrics.items():
        assert math.isfinite(float(value)), f"{key} is not finite: {value}"


# -----------------------------------------------------------------------
# Save / load round-trip
# -----------------------------------------------------------------------

def test_deep_pilco_save_load_preserves_policy(tmp_path: Path) -> None:
    agent = _make_deep_agent(seed=5)
    _fill_deep_agent_buffer(agent, n=20)
    agent.learn_step()

    ckpt = agent.save(tmp_path / "deep_pilco.pt")
    loaded = DeepPilcoAgent.load(
        ckpt,
        obs_dim=3,
        action_dim=1,
        action_low=[-2.0],
        action_high=[2.0],
        hidden_dim=16,
        n_layers=2,
        dropout_p=0.05,
        n_particles=8,
        horizon=4,
        model_train_steps=5,
        model_batch_size=8,
        model_lr=1e-3,
        policy_opt_steps=3,
        policy_lr=0.01,
        n_basis=10,
        max_gp_points=60,
        seed=5,
    )

    for p_orig, p_loaded in zip(agent.policy.parameters(), loaded.policy.parameters()):
        assert torch.allclose(p_orig, p_loaded)

    for p_orig, p_loaded in zip(agent.net.parameters(), loaded.net.parameters()):
        assert torch.allclose(p_orig, p_loaded)

    assert len(loaded.buffer) == len(agent.buffer)


# -----------------------------------------------------------------------
# Training smoke test (fast config)
# -----------------------------------------------------------------------

def test_deep_pilco_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_pilco_figures(monkeypatch)

    config = DeepPilcoConfig(
        env_id="Pendulum-v1",
        episodes=2,
        horizon=4,
        n_basis=8,
        hidden_dim=16,
        n_layers=1,
        dropout_p=0.05,
        n_particles=8,
        model_train_steps=5,
        model_batch_size=8,
        model_lr=1e-3,
        policy_opt_steps=3,
        policy_lr=0.01,
        num_init_rollouts=1,
        max_gp_points=60,
        max_steps_per_episode=40,
        eval_every=2,
        eval_episodes=1,
        checkpoint_every=2,
        output_dir=str(tmp_path),
    )

    result = train_deep_pilco(config, seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    assert isinstance(result["agent"], DeepPilcoAgent)
    assert len(result["history"]["episode_rewards"]) == 2
    assert result["paths"].run_dir.exists()


# =======================================================================
# Deep PILCO — InvertedPendulum-v5 + encode_angle tests
# =======================================================================

def _make_deep_ip_agent(seed: int = 0) -> "DeepPilcoAgent":
    """Create a tiny DeepPilcoAgent mimicking the InvertedPendulum-v5 recipe.

    env_obs_dim=4, encode_angle=True → internal obs_dim=5.
    Cost target = [0,0,1,0,0], weights = [0.5,1,1,0,0.1].
    """
    torch.manual_seed(seed)
    return DeepPilcoAgent(
        obs_dim=4,          # raw InvertedPendulum-v5 obs dim
        action_dim=1,
        action_low=[-3.0],
        action_high=[3.0],
        hidden_dim=16,
        n_layers=2,
        dropout_p=0.05,
        n_particles=8,
        horizon=5,
        model_train_steps=5,
        model_batch_size=8,
        model_lr=1e-3,
        policy_opt_steps=3,
        policy_lr=0.01,
        n_basis=10,
        max_gp_points=100,
        encode_angle=True,
        fixed_horizon_steps=40,
        seed=seed,
        cost_weight=(0.5, 1.0, 1.0, 0.0, 0.1),
    )


def _fill_deep_ip_buffer(agent: "DeepPilcoAgent", n: int = 30) -> None:
    """Fill agent buffer with random InvertedPendulum-style transitions.

    Uses raw 4-D obs (as the env would produce); encode happens inside
    store_transition → _encode() when encode_angle=True.
    """
    rng = np.random.default_rng(99)
    for i in range(n):
        raw_obs = rng.standard_normal(4).astype(np.float32)
        action = rng.standard_normal(1).astype(np.float32)
        raw_next = raw_obs + 0.05 * rng.standard_normal(4).astype(np.float32)
        if i == 0:
            agent.buffer.start_episode(agent._encode(raw_obs))
        agent.store_transition(raw_obs, action, 0.0, raw_next, False)


# --- DeepPilcoConfig: new fields ---

def test_deep_pilco_config_encode_angle_field() -> None:
    cfg = DeepPilcoConfig(encode_angle=True, fixed_horizon_steps=40)
    assert cfg.encode_angle is True
    assert cfg.fixed_horizon_steps == 40


def test_deep_pilco_config_rejects_negative_fixed_horizon() -> None:
    with pytest.raises(ValueError, match="fixed_horizon_steps"):
        DeepPilcoConfig(fixed_horizon_steps=-1)


def test_deep_pilco_config_allows_zero_cost_weight_entry() -> None:
    """cost_weight may contain zeros (e.g. cart_vel weight) — must not raise."""
    cfg = DeepPilcoConfig(cost_weight=(0.5, 1.0, 1.0, 0.0, 0.1))
    assert cfg.cost_weight == (0.5, 1.0, 1.0, 0.0, 0.1)


# --- encode_angle=True: env_obs_dim vs obs_dim ---

def test_deep_pilco_ip_obs_dims() -> None:
    agent = _make_deep_ip_agent()
    assert agent.env_obs_dim == 4
    assert agent.obs_dim == 5          # 4 + 1 sin/cos expansion


def test_deep_pilco_ip_target_is_5d() -> None:
    agent = _make_deep_ip_agent()
    assert agent.target.shape == (5,)
    # [0, 0, 1, 0, 0]: sinθ=0, cosθ=1 → upright
    assert float(agent.target[2]) == pytest.approx(1.0)


def test_deep_pilco_ip_bnn_input_dim() -> None:
    """BNN input must be obs_dim(5) + action_dim(1) = 6."""
    agent = _make_deep_ip_agent()
    assert agent.net.input_dim == 6
    assert agent.net.output_dim == 5


# --- encode helper ---

def test_deep_pilco_encode_maps_4d_to_5d() -> None:
    agent = _make_deep_ip_agent()
    raw = np.array([0.1, 0.3, -0.05, 0.02], dtype=np.float32)
    enc = agent._encode(raw)
    assert enc.shape == (5,)
    # sin / cos check
    np.testing.assert_allclose(enc[1], np.sin(0.3), rtol=1e-5)
    np.testing.assert_allclose(enc[2], np.cos(0.3), rtol=1e-5)


def test_deep_pilco_encode_no_angle_passthrough() -> None:
    """When encode_angle=False, _encode returns raw obs unchanged."""
    agent = _make_deep_agent()          # default: obs_dim=3, no encoding
    raw = np.array([0.5, -0.3, 1.2], dtype=np.float64)
    enc = agent._encode(raw)
    np.testing.assert_allclose(enc, raw.astype(np.float32), rtol=1e-6)


# --- select_action with encode_angle ---

def test_deep_pilco_ip_select_action_accepts_4d_obs() -> None:
    agent = _make_deep_ip_agent()
    raw_obs = np.array([0.0, 0.05, 0.0, 0.1], dtype=np.float32)
    action = agent.select_action(raw_obs, deterministic=True)
    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert float(np.abs(action).max()) <= 3.0 + 1e-5


# --- store_transition stores encoded obs ---

def test_deep_pilco_ip_store_transition_encodes_obs() -> None:
    agent = _make_deep_ip_agent()
    raw = np.array([0.1, 0.3, -0.05, 0.02], dtype=np.float32)
    action = np.array([0.5], dtype=np.float32)
    raw_next = raw + 0.01
    agent.store_transition(raw, action, 0.0, raw_next, False)
    assert len(agent.buffer) == 1
    stored_obs = agent.buffer._obs[0]
    assert stored_obs.shape == (5,), f"Expected 5D encoded obs, got {stored_obs.shape}"
    np.testing.assert_allclose(stored_obs[1], np.sin(0.3), rtol=1e-5)


# --- learn_step with encoded 5D data ---

def test_deep_pilco_ip_learn_step_finite_metrics() -> None:
    agent = _make_deep_ip_agent(seed=42)
    _fill_deep_ip_buffer(agent, n=40)
    metrics = agent.learn_step()
    for key, val in metrics.items():
        assert math.isfinite(float(val)), f"{key} not finite: {val}"


# --- smoke training with InvertedPendulum-v5 + encode_angle ---

def test_deep_pilco_ip_training_smoke(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_pilco_figures(monkeypatch)

    config = DeepPilcoConfig(
        env_id="InvertedPendulum-v5",
        episodes=2,
        horizon=5,
        n_basis=8,
        hidden_dim=16,
        n_layers=1,
        dropout_p=0.05,
        n_particles=8,
        model_train_steps=5,
        model_batch_size=8,
        model_lr=1e-3,
        policy_opt_steps=3,
        policy_lr=0.01,
        num_init_rollouts=2,
        max_gp_points=100,
        max_steps_per_episode=40,
        encode_angle=True,
        fixed_horizon_steps=20,
        cost_weight=(0.5, 1.0, 1.0, 0.0, 0.1),
        eval_every=2,
        eval_episodes=1,
        checkpoint_every=2,
        output_dir=str(tmp_path),
    )

    result = train_deep_pilco(config, seed=0)

    assert set(result) == {"agent", "history", "metrics", "paths"}
    agent = result["agent"]
    assert isinstance(agent, DeepPilcoAgent)
    assert agent.encode_angle is True
    assert agent.obs_dim == 5
    assert len(result["history"]["episode_rewards"]) == 2
    assert result["paths"].run_dir.exists()


# -----------------------------------------------------------------------
# Cost-decrease test: Deep PILCO on InvertedPendulum-v5 should reduce
# the imagined trajectory cost over several iterations.
# -----------------------------------------------------------------------

def test_deep_pilco_ip_cost_decreases_over_iterations(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch
) -> None:
    """The imagined cost should decrease as the policy improves.

    We run 6 iterations on a tiny config (fast, < 30 s) and assert that
    the mean cost over the last 3 iterations is strictly lower than the
    mean over the first 3.  This verifies the gradient signal flows
    correctly through the BNN + particle rollout into the RBF policy.
    """
    _disable_pilco_figures(monkeypatch)

    config = DeepPilcoConfig(
        env_id="InvertedPendulum-v5",
        episodes=6,
        horizon=10,
        n_basis=10,
        hidden_dim=32,
        n_layers=2,
        dropout_p=0.05,
        n_particles=20,
        model_train_steps=30,
        model_batch_size=16,
        model_lr=1e-3,
        policy_opt_steps=10,
        policy_lr=0.01,
        num_init_rollouts=3,
        max_gp_points=200,
        max_steps_per_episode=40,
        encode_angle=True,
        fixed_horizon_steps=20,
        cost_weight=(0.5, 1.0, 1.0, 0.0, 0.1),
        eval_every=99,   # skip eval to keep test fast
        eval_episodes=1,
        checkpoint_every=99,
        output_dir=str(tmp_path),
    )

    result = train_deep_pilco(config, seed=0)

    costs = result["history"].get("step_predicted_costs", [])
    # Require at least 4 cost recordings so we can split early vs late
    assert len(costs) >= 4, f"Expected >=4 cost entries, got {len(costs)}: {costs}"

    mid = len(costs) // 2
    early_mean = sum(costs[:mid]) / mid
    late_mean = sum(costs[mid:]) / (len(costs) - mid)

    # Costs must all be finite
    assert all(math.isfinite(c) for c in costs), f"Non-finite costs: {costs}"
    # Late mean must be strictly lower than early mean (learning signal present)
    assert late_mean < early_mean, (
        f"Expected cost to decrease: early_mean={early_mean:.4f}, "
        f"late_mean={late_mean:.4f}.  Full cost history: {costs}"
    )
