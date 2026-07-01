"""Reporting for PETS experiments.

Generates post-training figures:
- Learning curve (episode reward + moving average + eval).
- Dynamics NLL over episodes.
- Ensemble disagreement (epistemic uncertainty) over episodes.
- Buffer size growth over episodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import BaseReporting


class PetsReporting(BaseReporting):
    """Post-training figures for PETS."""

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate all PETS training figures.

        Calls the base class for the learning-curve figure, then adds
        PETS-specific metrics: dynamics NLL, ensemble disagreement, and
        buffer size growth.

        Parameters
        ----------
        history:
            Training history dict from ``RunRecorder``.
        config:
            ``PetsConfig`` instance (used for title strings).
        output_dir:
            Root run directory; figures are saved in ``output_dir/figures/``.

        Returns
        -------
        list[Path]
            Paths to all created PNG files.
        """
        # Base: learning curve (episode reward)
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"
        approach = getattr(config, "approach", "pets")

        # Dynamics NLL over episodes
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_dynamics_nlls",
            name="dynamics_nll",
            title=f"Ensemble dynamics NLL — {approach}",
            filename="dynamics_nll.png",
        )

        # Ensemble disagreement (epistemic uncertainty)
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_ensemble_disagreements",
            name="ensemble_disagreement",
            title=f"Ensemble disagreement (epistemic σ) — {approach}",
            filename="ensemble_disagreement.png",
        )

        # Buffer size over episodes
        self._maybe_plot_metric(
            figures, history, fig_dir,
            key="step_buffer_sizes",
            name="buffer_size",
            title=f"Replay buffer size — {approach}",
            filename="buffer_size.png",
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
    return PetsReporting().generate_figures(history, config, run_dir)
