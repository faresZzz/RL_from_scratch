"""Reporting for Action-JEPA runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting


class ActionJepaReporting(BaseReporting):
    """Post-training figures for world-model quality and collapse monitoring."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "action_jepa")

        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_representation_prediction_losses",
            name="representation_prediction_loss",
            title=f"Phase-A masked JEPA loss — {approach}",
            filename="representation_prediction_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_representation_variance_losses",
            name="representation_variance_loss",
            title=f"Phase-A variance loss — {approach}",
            filename="representation_variance_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_representation_covariance_losses",
            name="representation_covariance_loss",
            title=f"Phase-A covariance loss — {approach}",
            filename="representation_covariance_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_effective_ranks",
            name="effective_rank",
            title=f"Phase-A effective rank — {approach}",
            filename="effective_rank.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_latent_prediction_losses",
            name="latent_prediction_loss",
            title=f"One-step latent prediction loss — {approach}",
            filename="latent_prediction_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_rollout_prediction_losses",
            name="rollout_prediction_loss",
            title=f"Multi-step latent rollout loss — {approach}",
            filename="rollout_prediction_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_reward_prediction_losses",
            name="reward_prediction_loss",
            title=f"Reward head loss — {approach}",
            filename="reward_prediction_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_continuation_losses",
            name="continuation_loss",
            title=f"Continuation head loss — {approach}",
            filename="continuation_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_variance_losses",
            name="variance_loss",
            title=f"VICReg variance loss — {approach}",
            filename="variance_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_covariance_losses",
            name="covariance_loss",
            title=f"VICReg covariance loss — {approach}",
            filename="covariance_loss.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_latent_stds",
            name="latent_std",
            title=f"Collapse monitor — {approach}",
            filename="latent_std.png",
        )
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_collapse_gaps",
            name="collapse_gap",
            title=f"Collapse gap — {approach}",
            filename="collapse_gap.png",
        )
        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    run_dir: Path,
) -> list[Path]:
    return ActionJepaReporting().generate_figures(history, config, run_dir)
