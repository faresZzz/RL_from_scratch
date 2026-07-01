"""Reporting for Dyna-family experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting, plot_generic_metric


class DynaReporting(BaseReporting):
    """Post-training figures for tabular and deep dyna variants."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "dyna")

        losses = history.get("step_q_losses", [])
        if losses:
            figures.append(
                self.plot_loss_curves(
                    {"q_loss": losses},
                    title=f"Q loss — {approach}",
                    output_path=fig_dir / "q_loss.png",
                )
            )

        epsilons = history.get("epsilons", [])
        if epsilons:
            figures.append(
                self.plot_epsilon_decay(
                    epsilons,
                    title=f"Epsilon — {approach}",
                    output_path=fig_dir / "epsilon_decay.png",
                )
            )

        metric_labels = {
            "step_real_td_errors": "real_td_error",
            "step_planning_td_errors": "planning_td_error",
            "step_model_buffer_sizes": "model_buffer_size",
            "step_model_prediction_losses": "model_prediction_loss",
            "step_reward_prediction_losses": "reward_prediction_loss",
            "step_done_prediction_losses": "done_prediction_loss",
            "step_exploration_bonuses": "exploration_bonus",
            "step_imagined_update_counts": "imagined_update_count",
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
    output_dir: Path,
) -> list[Path]:
    return DynaReporting().generate_figures(history, config, output_dir)
