"""Bayesian Neural Network dynamics + particle propagation for Deep PILCO.

Deep PILCO (Gal et al., 2016) replaces the GP dynamics model of PILCO with a
Bayesian neural network (BNN) approximated via **MC-dropout** (Gal & Ghahramani,
2016).  The key ideas are:

1. **MC-dropout as Bayesian approximation**: applying dropout *at test time* gives
   approximate samples from the BNN posterior.  Each forward pass with a different
   dropout mask samples a plausible dynamics function.

2. **Temporally-correlated masks**: in classical PILCO, a single GP samples a
   *consistent* dynamics function.  Deep PILCO mirrors this by assigning **each
   particle a fixed dropout mask** that is kept constant over the entire planning
   horizon.  The particle thus experiences a coherent, single-sampled dynamics
   function throughout its trajectory — rather than an independent noise draw at
   each step.  This correlation is essential: independent masks per step would
   underestimate trajectory uncertainty by averaging out errors that compound in
   real rollouts.

3. **Particle propagation with moment matching**: instead of analytic Gaussian
   belief propagation (which is intractable for a neural network), we maintain a
   *particle cloud* of K states.  After each step we **moment-match** the cloud
   (fit an empirical Gaussian) and **resample** K fresh particles from it.  The
   moment-matching step keeps the distribution Gaussian and tractable, matching the
   analytic propagation philosophy of the original PILCO.

Algorithm (one planning horizon):
    μ₀, Σ₀   initial state belief
    {xₖ}   ~ N(μ₀, Σ₀)          sample K particles
    mask_k   fixed per particle   sample dropout masks once
    for t = 1 … H:
        aₖ  = π(xₖ)              RBF policy (deterministic per particle)
        Δxₖ = f_θ(xₖ, aₖ; maskₖ)  BNN forward with fixed mask
        x'ₖ = xₖ + Δxₖ           next state
        cost_t = mean_k c(x'ₖ)   particle-averaged saturating cost
        xₖ  = resample(x'ₖ)      moment-matching resample
    J = Σ_t cost_t               total cost (differentiable w.r.t. π)

References
----------
- Gal, Y., McAllister, R., & Rasmussen, C.E. (2016).
  *Improving PILCO with Bayesian Neural Network Dynamics Models.*
  Data-Efficient Machine Learning Workshop, ICML 2016.
- Gal, Y. & Ghahramani, Z. (2016).
  *Dropout as a Bayesian Approximation.*
  ICML 2016.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl_from_scratch.pilco.cost import saturating_cost
from rl_from_scratch.pilco.policy import RBFPolicy
from rl_from_scratch.pilco.belief_propagation import project_encoded_angle_torch


# ======================================================================
# Bayesian Dynamics Network (MC-dropout MLP)
# ======================================================================

class BayesianDynamicsNetwork(nn.Module):
    """MLP dynamics model with MC-dropout for approximate Bayesian inference.

    The network maps ``(state, action)`` → ``Δstate`` (state delta).  Dropout
    is implemented **explicitly via Bernoulli masks** rather than ``nn.Dropout``
    so that a particle can supply a *fixed* mask and see a consistent dynamics
    function over the entire planning horizon (temporally-correlated dropout).

    Standard training with ``model.train()`` samples fresh masks; planning with
    ``model.eval()`` still samples masks but the caller can provide a fixed set.

    Parameters
    ----------
    input_dim:
        Dimension of the concatenated ``(state, action)`` input.
    output_dim:
        State dimension (same as ``state_dim``).
    hidden_dim:
        Width of each hidden layer.
    n_layers:
        Number of hidden layers (≥ 1).
    dropout_p:
        Dropout probability ``p`` (fraction of units dropped).  The active
        units are scaled by ``1 / (1 - p)`` to keep the expected activation
        magnitude constant (inverted-dropout convention).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1).")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout_p = float(dropout_p)

        # --- Build linear layers ---
        dims = [input_dim] + [hidden_dim] * n_layers
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Inverted-dropout scale: 1 / (1 - p)
        self._scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 1.0

    # ------------------------------------------------------------------
    # Mask sampling
    # ------------------------------------------------------------------

    def sample_masks(self, n_particles: int, device: torch.device | None = None) -> list[Tensor]:
        """Sample one Bernoulli dropout mask per hidden layer for ``n_particles``.

        Returns a list of ``n_layers`` tensors, each of shape
        ``[n_particles, hidden_dim]``, containing 0/1 values.  The caller
        holds onto these and passes them unchanged to every ``forward`` call
        during the planning horizon so each particle sees a consistent
        dynamics function.

        Masks are sampled with ``p_keep = 1 - dropout_p`` Bernoulli draws.
        """
        p_keep = 1.0 - self.dropout_p
        masks = []
        for _ in range(self.n_layers):
            m = torch.bernoulli(
                torch.full((n_particles, self.hidden_dim), p_keep, device=device)
            )
            masks.append(m)
        return masks

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, xu: Tensor, masks: list[Tensor] | None = None) -> Tensor:
        """Predict state delta ``Δx = f_θ(xu ; masks)``.

        Parameters
        ----------
        xu:
            Input ``[B, input_dim]`` — concatenated state and action.
        masks:
            List of ``n_layers`` binary tensors each ``[B, hidden_dim]``.
            When ``None`` (e.g. standard training), fresh masks are sampled
            internally so dropout behaves like standard stochastic dropout.

        Returns
        -------
        Δx : Tensor
            Predicted state delta ``[B, output_dim]``.
        """
        B = xu.shape[0]
        if masks is None:
            # Standard stochastic forward: sample fresh masks (train or eval)
            masks = self.sample_masks(B, device=xu.device)

        h = xu
        for layer_idx, linear in enumerate(self.linears):
            h = linear(h)
            h = F.silu(h)
            # Apply scaled Bernoulli mask (inverted dropout)
            m = masks[layer_idx].to(h.dtype).to(h.device)   # [B, hidden_dim]
            h = h * m * self._scale

        return self.output_layer(h)   # [B, output_dim]


# ======================================================================
# One-step particle propagation
# ======================================================================

def propagate_particles(
    net: BayesianDynamicsNetwork,
    policy: torch.nn.Module,
    particles: Tensor,
    masks: list[Tensor],
    target: Tensor,
    weight: Tensor,
    action_high: Tensor,
    project_encoded_angle: bool = False,
    step_cost_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Propagate a particle cloud one step forward under BNN dynamics + RBF policy.

    Algorithm
    ---------
    For each particle k = 1 … K:
        aₖ = π(xₖ)                    RBF policy; already bounded by sine saturation
        Δxₖ = f_θ(xₖ ⊕ aₖ ; maskₖ)   BNN forward with particle-specific mask
        x'ₖ = xₖ + Δxₖ                residual next state

    Then:
        step_cost = (1/K) Σₖ c(x'ₖ)   mean saturating cost over particles
        moment-match: μ = mean(x'ₖ), Σ = cov(x'ₖ)
        resample K fresh x''ₖ ~ N(μ, Σ)   (restores Gaussian form)

    The resampling step is the Deep PILCO "moment-matching" analogue of the
    analytic Gaussian propagation in classical PILCO.  It keeps the particle
    cloud Gaussian at every step, making the representation tractable across the
    whole horizon while still using a non-linear BNN dynamics model.

    Parameters
    ----------
    net:
        Bayesian dynamics network.
    policy:
        RBF policy (any dtype; handles batched ``[K, state_dim]`` inputs).
    particles:
        Current particle cloud ``[K, state_dim]``.
    masks:
        Fixed per-particle dropout masks — list of ``n_layers`` tensors each
        ``[K, hidden_dim]``.  Passed directly to ``net.forward``.
    target:
        Target state ``[state_dim]`` for the saturating cost.
    weight:
        Precision matrix ``[state_dim, state_dim]`` for the saturating cost.
    action_high:
        Not used directly (RBFPolicy handles saturation internally) but kept
        as a parameter for signature consistency with the trajectory function.

    Returns
    -------
    next_particles:
        Resampled particle cloud ``[K, state_dim]`` (gradient-connected).
    step_cost:
        Scalar mean saturating cost over the un-resampled ``x'ₖ``.
    """
    del action_high  # RBFPolicy already saturates to ±action_high internally
    K = particles.shape[0]

    # 1. Compute actions for all particles in one batched pass
    actions = policy.forward(particles)             # [K, action_dim]

    # 2. BNN forward with temporally-correlated, per-particle masks
    xu = torch.cat([particles, actions], dim=-1)    # [K, state_dim + action_dim]
    delta = net.forward(xu, masks)                  # [K, state_dim]

    # 3. Residual next state
    next_p = particles + delta                      # [K, state_dim]
    if project_encoded_angle:
        # Keep imagined encoded angles physically valid: sin²(theta)+cos²(theta)=1.
        next_p = project_encoded_angle_torch(next_p)

    # 4. Step cost = mean saturating cost over the particle cloud
    # saturating_cost accepts [..., D]; cost shape [K]
    costs = (
        saturating_cost(next_p, target, weight)
        if step_cost_fn is None
        else step_cost_fn(next_p, actions)
    )
    step_cost = costs.mean()                        # scalar

    # 5. Moment-matching resample: fit N(μ, Σ) to the cloud and redraw K particles
    #    Gradient flows through next_p -> mu and the Cholesky factor.
    mu = next_p.mean(dim=0)                         # [state_dim]
    # Unbiased empirical covariance; clamp for numerical stability
    centered = next_p - mu.unsqueeze(0)             # [K, D]
    sigma = (centered.t() @ centered) / max(K - 1, 1)    # [D, D]
    sigma = sigma + 1e-5 * torch.eye(sigma.shape[0], dtype=sigma.dtype, device=sigma.device)

    # Cholesky resample: x'' = μ + ε L^T, ε ~ N(0, I)
    chol = torch.linalg.cholesky(sigma)             # [D, D]
    eps = torch.randn(K, mu.shape[0], dtype=mu.dtype, device=mu.device, generator=generator)
    next_particles = mu.unsqueeze(0) + eps @ chol.t()     # [K, D]
    if project_encoded_angle:
        # The Gaussian resample is unconstrained; project again before the next imagined step.
        next_particles = project_encoded_angle_torch(next_particles)

    return next_particles, step_cost


# ======================================================================
# Full horizon particle trajectory prediction
# ======================================================================

def predict_trajectory_particles(
    net: BayesianDynamicsNetwork,
    policy: torch.nn.Module,
    mu0: Tensor | None = None,
    sigma0: Tensor | None = None,
    *,
    particles0: Tensor | None = None,
    masks: list[Tensor] | None = None,
    horizon: int,
    target: Tensor,
    weight: Tensor,
    n_particles: int = 30,
    action_high: Tensor | None = None,
    project_encoded_angle: bool = False,
    step_cost_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """Roll K particles forward ``horizon`` steps accumulating the expected cost.

    This is the Deep PILCO objective ``J(π)`` that Adam minimises to improve
    the policy.  The objective is the mean cost per planning step.  The entire
    computation is differentiable w.r.t. the policy parameters, so
    ``J.backward()`` flows gradients through the chain:

        resample → policy → BNN → saturating_cost → J

    Because the BNN masks are **fixed per particle for the whole horizon**,
    the imagined trajectories are correlated across time — each particle
    "experiences" one consistent sample of the dynamics function.  This
    makes the cost estimate a more faithful proxy for actual trajectory
    uncertainty than independent-mask approaches.

    Parameters
    ----------
    net:
        ``BayesianDynamicsNetwork`` in any state (train or eval; masks are
        managed explicitly so the training/eval mode does not affect dropout).
    policy:
        Differentiable bounded policy whose parameters we are optimising.
    mu0:
        Initial state belief mean ``[state_dim]``.
    sigma0:
        Initial state belief covariance ``[state_dim, state_dim]`` (PD).
    horizon:
        Number of propagation steps H.
    target:
        Target state ``[state_dim]`` for the saturating cost.
    weight:
        Precision matrix ``[state_dim, state_dim]`` for the saturating cost.
    n_particles:
        Number of particles K.
    action_high:
        Passed through to ``propagate_particles`` (unused internally).

    Returns
    -------
    total_cost:
        Scalar mean per-step cost
        ``J = (1/H) Σ_{t=1}^{H} cost_t`` — the policy optimisation objective.
    mean_trajectory:
        List of ``horizon + 1`` particle-mean state tensors
        ``[state_dim]`` (detached, for diagnostics).
    """
    if particles0 is None:
        if mu0 is None or sigma0 is None:
            raise ValueError("Provide either particles0 or both mu0 and sigma0.")
        sigma0_stable = sigma0 + 1e-5 * torch.eye(
            sigma0.shape[0], dtype=sigma0.dtype, device=sigma0.device
        )
        chol0 = torch.linalg.cholesky(sigma0_stable)
        eps0 = torch.randn(
            n_particles,
            mu0.shape[0],
            dtype=mu0.dtype,
            device=mu0.device,
            generator=generator,
        )
        particles = mu0.unsqueeze(0) + eps0 @ chol0.t()
    else:
        particles = particles0
        n_particles = particles.shape[0]
        if mu0 is None:
            mu0 = particles.mean(dim=0)
    if action_high is None:
        action_high = torch.ones(1, dtype=particles.dtype, device=particles.device)

    # --- Sample FIXED per-particle dropout masks for the whole horizon ---
    #     Each particle keeps its mask unchanged across all H steps.
    if masks is None:
        masks = net.sample_masks(n_particles, device=particles.device)

    # Cast masks to the same dtype as the particles (float32 or float64)
    masks = [m.to(particles.dtype) for m in masks]

    total_cost = torch.zeros((), dtype=particles.dtype, device=particles.device)
    mean_trajectory: list[Tensor] = [particles.mean(dim=0).detach()]

    for _ in range(horizon):
        particles, step_cost = propagate_particles(
            net,
            policy,
            particles,
            masks,
            target,
            weight,
            action_high,
            project_encoded_angle=project_encoded_angle,
            step_cost_fn=step_cost_fn,
            generator=generator,
        )
        total_cost = total_cost + step_cost
        mean_trajectory.append(particles.mean(dim=0).detach())

    return total_cost / max(horizon, 1), mean_trajectory


# ======================================================================
# BNN training helper
# ======================================================================

def train_bnn_on_buffer(
    net: BayesianDynamicsNetwork,
    X: Tensor,
    Y: Tensor,
    *,
    n_steps: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Train the BNN dynamics on buffered ``(state, action, Δstate)`` data.

    Each minibatch forward pass uses *fresh* masks (standard stochastic
    dropout), which acts as a regulariser.  MSE loss on the predicted delta.

    Parameters
    ----------
    net:
        ``BayesianDynamicsNetwork``; put in train mode internally.
    X:
        Input tensor ``[N, state_dim + action_dim]`` (float32 or float64).
    Y:
        Target delta tensor ``[N, state_dim]``.
    n_steps:
        Number of gradient steps.
    batch_size:
        Minibatch size; clamped to ``len(X)`` if smaller.
    lr:
        Adam learning rate.
    seed:
        Seed controlling the 80/20 split and minibatch sampling.

    Returns
    -------
    mean_loss:
        Mean MSE loss over the last 10% of steps (finite float).
    metrics:
        Train/validation losses plus split metadata and optimizer knobs.
    """
    # Cast data to match network dtype
    sample_p = next(net.parameters())
    dtype = sample_p.dtype
    device = sample_p.device
    X = X.to(dtype=dtype, device=device)
    Y = Y.to(dtype=dtype, device=device)

    N = X.shape[0]
    batch_size = min(batch_size, N)

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(N, generator=gen, device=device)
    split = max(1, int(0.8 * N))
    train_idx = perm[:split]
    val_idx = perm[split:] if split < N else perm[:split]
    weight_decay = 1e-5
    grad_clip = 10.0
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    losses: list[float] = []

    for step in range(n_steps):
        batch_size_eff = min(batch_size, len(train_idx))
        choice = torch.randint(0, len(train_idx), (batch_size_eff,), generator=gen, device=device)
        batch_idx = train_idx[choice]
        xb = X[batch_idx]
        yb = Y[batch_idx]

        # Standard stochastic dropout (fresh masks sampled inside forward)
        pred = net.forward(xb, masks=None)
        loss = F.mse_loss(pred, yb)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        val = float(loss.item())
        if math.isfinite(val):
            losses.append(val)

    net.eval()
    tail = losses[-max(1, len(losses) // 10) :] if losses else [1.0]
    mean_loss = float(sum(tail) / len(tail))
    with torch.no_grad():
        train_loss = float(F.mse_loss(net(X[train_idx]), Y[train_idx]).item())
        val_loss = float(F.mse_loss(net(X[val_idx]), Y[val_idx]).item())
    return mean_loss, {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "seed": seed,
    }
