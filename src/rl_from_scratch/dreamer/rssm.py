"""Recurrent State-Space Model (RSSM) for DreamerV1.

The RSSM maintains a pair of state representations:
- Deterministic (recurrent) state ``h`` via a GRU cell.
- Stochastic state ``z`` sampled from a diagonal Gaussian.

Two types of transitions:
- ``obs_step`` (posterior): conditions on the current observation embedding
  to compute both the prior (from ``h`` alone) and the posterior
  (from ``h + embed``).  The posterior is used during learning.
- ``img_step`` (prior-only): imagines the next prior state from the
  previous state + action, without observation.  Used during behaviour
  optimisation in imagination.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import softplus


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    Parameters
    ----------
    action_dim:
        Dimensionality of the action space.
    embed_dim:
        Dimensionality of the observation embedding (from ``Encoder``).
    deter_dim:
        Dimensionality of the deterministic (GRU hidden) state.
    stoch_dim:
        Dimensionality of the stochastic latent state.
    hidden_dim:
        Width of the MLP heads that predict the Gaussian stats.
    min_std:
        Minimum standard deviation (added after softplus) to keep
        distributions well-conditioned.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        # Input projection: cat(z_{t-1}, a_{t-1}) → pre-GRU input
        self.in_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.SiLU(),
        )
        # Recurrent cell
        self.cell = nn.GRUCell(hidden_dim, deter_dim)

        # Prior: h_t → (μ_prior, σ_prior)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
        )
        self.prior_stats = nn.Linear(hidden_dim, 2 * stoch_dim)

        # Posterior: cat(h_t, embed_t) → (μ_post, σ_post)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.SiLU(),
        )
        self.post_stats = nn.Linear(hidden_dim, 2 * stoch_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def initial_state(self, batch: int, device: torch.device | str = "cpu") -> dict:
        """Return a zero-initialised RSSM state dict for *batch* particles."""
        z = torch.zeros(batch, self.stoch_dim, device=device)
        h = torch.zeros(batch, self.deter_dim, device=device)
        return {
            "deter": h,
            "stoch": z,
            "mean": z.clone(),
            "std": torch.ones(batch, self.stoch_dim, device=device),
        }

    def _dist_stats(
        self,
        net: nn.Module,
        stats_layer: nn.Module,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (mean, std) from a net + stats_layer on input x."""
        h = net(x)
        raw = stats_layer(h)
        mean, raw_std = raw.chunk(2, dim=-1)
        std = softplus(raw_std) + self.min_std
        return mean, std

    # ------------------------------------------------------------------
    # Core transitions
    # ------------------------------------------------------------------

    def obs_step(
        self,
        prev: dict,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[dict, dict]:
        """Posterior transition — conditions on the observation embedding.

        Parameters
        ----------
        prev:
            Previous RSSM state dict (keys: ``deter``, ``stoch``).
        prev_action:
            Previous action tensor ``[B, action_dim]``.
        embed:
            Observation embedding ``[B, embed_dim]`` from the Encoder.

        Returns
        -------
        (post, prior)
            Both are RSSM state dicts with keys
            ``deter``, ``stoch``, ``mean``, ``std``.
        """
        # GRU step: h_t = f(h_{t-1}, cat(z_{t-1}, a_{t-1}))
        x = self.in_proj(torch.cat([prev["stoch"], prev_action], dim=-1))
        deter = self.cell(x, prev["deter"])

        # Prior: p(z_t | h_t)
        pm, ps = self._dist_stats(self.prior_net, self.prior_stats, deter)
        prior = {
            "deter": deter,
            "stoch": pm + ps * torch.randn_like(ps),
            "mean": pm,
            "std": ps,
        }

        # Posterior: q(z_t | h_t, e_t)
        qm, qs = self._dist_stats(
            self.post_net, self.post_stats, torch.cat([deter, embed], dim=-1)
        )
        post = {
            "deter": deter,
            "stoch": qm + qs * torch.randn_like(qs),
            "mean": qm,
            "std": qs,
        }

        return post, prior

    def img_step(self, prev: dict, prev_action: torch.Tensor) -> dict:
        """Prior transition — imagination step (no observation).

        Parameters
        ----------
        prev:
            Previous RSSM state dict.
        prev_action:
            Previous action tensor ``[B, action_dim]``.

        Returns
        -------
        dict
            Prior RSSM state (keys: ``deter``, ``stoch``, ``mean``, ``std``).
        """
        x = self.in_proj(torch.cat([prev["stoch"], prev_action], dim=-1))
        deter = self.cell(x, prev["deter"])

        pm, ps = self._dist_stats(self.prior_net, self.prior_stats, deter)
        return {
            "deter": deter,
            "stoch": pm + ps * torch.randn_like(ps),
            "mean": pm,
            "std": ps,
        }

    # ------------------------------------------------------------------
    # Feature extractor
    # ------------------------------------------------------------------

    def get_feat(self, state: dict) -> torch.Tensor:
        """Concatenate deterministic and stochastic state to form the feature.

        Returns a tensor of shape ``[..., deter_dim + stoch_dim]``.
        """
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    # ------------------------------------------------------------------
    # KL loss
    # ------------------------------------------------------------------

    def kl_loss(
        self,
        post: dict,
        prior: dict,
        free_nats: float,
    ) -> torch.Tensor:
        """KL divergence between posterior and prior, clamped by free_nats.

        KL( q(z|h,e) || p(z|h) ) summed over the stoch dimension then
        averaged over the batch.

        Parameters
        ----------
        post:
            Posterior state dict with keys ``mean``, ``std``.
        prior:
            Prior state dict with keys ``mean``, ``std``.
        free_nats:
            Lower bound on KL (prevents the model from collapsing the
            stochastic state before the posterior can latch on).

        Returns
        -------
        torch.Tensor
            Scalar KL loss.
        """
        q = torch.distributions.Normal(post["mean"], post["std"])
        p = torch.distributions.Normal(prior["mean"], prior["std"])
        kl = torch.distributions.kl_divergence(q, p).sum(dim=-1)
        return torch.clamp(kl, min=free_nats).mean()
