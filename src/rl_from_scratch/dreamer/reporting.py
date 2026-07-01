"""Reporting for DreamerV1 experiments.

Generates post-training figures:
- Learning curve (episode reward + moving average + eval).
- World-model losses: reconstruction, reward prediction, KL.
- Behaviour losses: actor, critic.
- Imagined return over training steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting


class DreamerReporting(BaseReporting):
    """Post-training figures for DreamerV1."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate all Dreamer training figures.

        Calls the base class for the learning-curve, then adds
        Dreamer-specific losses and metrics.

        Parameters
        ----------
        history:
            Training history dict from ``RunRecorder``.
        config:
            ``DreamerConfig`` instance (used for title strings).
        output_dir:
            Root run directory; figures are saved in ``output_dir/figures/``.

        Returns
        -------
        list[Path]
            Paths to all created PNG files.
        """
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "dreamer")

        # World-model losses
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_recon_losses",
            name="recon_loss",
            title=f"Reconstruction loss — {approach}",
            filename="recon_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_reward_losses",
            name="reward_loss",
            title=f"Reward prediction loss — {approach}",
            filename="reward_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_kls",
            name="kl",
            title=f"KL divergence (free-nats clamped) — {approach}",
            filename="kl.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_model_losses",
            name="model_loss",
            title=f"World-model total loss — {approach}",
            filename="model_loss.png",
        )

        # Behaviour losses
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_actor_losses",
            name="actor_loss",
            title=f"Actor loss — {approach}",
            filename="actor_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_critic_losses",
            name="critic_loss",
            title=f"Critic loss — {approach}",
            filename="critic_loss.png",
        )

        # Imagined return
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_imagined_returns",
            name="imagined_return",
            title=f"Imagined λ-return — {approach}",
            filename="imagined_return.png",
        )

        # Episode lengths
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="episode_lengths",
            name="episode_length",
            title=f"Episode length — {approach}",
            filename="episode_length.png",
        )

        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    run_dir: Path,
) -> list[Path]:
    """Module-level entry-point called by ``RunManager.finalize_run``."""
    return DreamerReporting().generate_figures(history, config, run_dir)
