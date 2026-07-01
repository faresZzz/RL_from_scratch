"""Reporting for MBPO experiments.

Generates post-training figures:
- Learning curve (episode reward + moving average + eval).
- Model NLL over training steps.
- Ensemble disagreement (epistemic uncertainty) over steps.
- Model buffer size growth over steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting


class MbpoReporting(BaseReporting):
    """Post-training figures for MBPO."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate all MBPO training figures.

        Calls the base class for the learning-curve figure, then adds
        MBPO-specific metrics: model NLL, ensemble disagreement, and
        model buffer size.

        Parameters
        ----------
        history:
            Training history dict from ``RunRecorder``.
        config:
            ``MbpoConfig`` instance (used for title strings).
        output_dir:
            Root run directory; figures are saved in ``output_dir/figures/``.

        Returns
        -------
        list[Path]
            Paths to all created PNG files.
        """
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "mbpo")

        # Model NLL over training updates
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_model_nlls",
            name="model_nll",
            title=f"Ensemble dynamics NLL — {approach}",
            filename="model_nll.png",
        )

        # Ensemble disagreement (epistemic uncertainty)
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_ensemble_disagreements",
            name="ensemble_disagreement",
            title=f"Ensemble disagreement (epistemic σ) — {approach}",
            filename="ensemble_disagreement.png",
        )

        # Model buffer size over steps
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_model_buffer_sizes",
            name="model_buffer_size",
            title=f"Model buffer size — {approach}",
            filename="model_buffer_size.png",
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
    return MbpoReporting().generate_figures(history, config, run_dir)
