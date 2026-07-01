"""Reporting helpers for MuZero experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting, plot_generic_metric


class MuZeroReporting(BaseReporting):
    """Post-training figures for MuZero."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "muzero")
        metric_labels = {
            "step_losses": "loss",
            "step_policy_losses": "policy_loss",
            "step_value_losses": "value_loss",
            "step_reward_losses": "reward_loss",
            "step_root_value_means": "root_value_mean",
        }

        for metric_name, label in metric_labels.items():
            values = history.get(metric_name, [])
            if not values:
                continue
            figures.append(
                plot_generic_metric(
                    values,
                    name=label,
                    title=f"{label} — {approach}",
                    output_path=fig_dir / f"{label}.png",
                )
            )
        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    run_dir: Path,
) -> list[Path]:
    return MuZeroReporting().generate_figures(history, config, run_dir)
