"""Reporting for PILCO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting


class PilcoReporting(BaseReporting):
    """Post-training figures for PILCO."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        # Base: learning curves (episode reward)
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "pilco")

        # Predicted cost over iterations
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_predicted_costs",
            name="predicted_cost",
            title=f"Predicted trajectory cost — {approach}",
            filename="predicted_cost.png",
        )

        # GP model NLL over iterations
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_model_nlls",
            name="model_nll",
            title=f"GP marginal likelihood (NLML) — {approach}",
            filename="model_nll.png",
        )

        # Number of GP training points over iterations
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_gp_points",
            name="gp_points",
            title=f"GP training points — {approach}",
            filename="gp_points.png",
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
    return PilcoReporting().generate_figures(history, config, run_dir)
