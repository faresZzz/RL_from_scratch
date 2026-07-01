"""Post-training report for the REINFORCE agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import (
    BaseReporting,
)


class ReinforceReporting(BaseReporting):
    """Post-training figures for the REINFORCE agents.

    Produces, in addition to the base learning curve:
    - Loss curves (if available in step_losses).
    """

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the figures for the REINFORCE agents.

        Parameters
        ----------
        history:
            Dictionary containing ``episode_rewards`` and possibly
            ``step_losses``.
        config:
            Experiment configuration.
        output_dir:
            Root directory of the run.

        Returns
        -------
        list[Path]
            Paths of the created PNGs.
        """
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"

        approach = getattr(config, "approach", "reinforce")

        loss_series: dict[str, list[float]] = {}
        policy_losses = history.get("step_policy_losses", [])
        value_losses = history.get("step_value_losses", [])
        if policy_losses:
            loss_series["policy_loss"] = policy_losses
        if value_losses:
            loss_series["value_loss"] = value_losses
        if loss_series:
            path = self.plot_loss_curves(
                loss_series,
                title=f"Losses — {approach}",
                output_path=fig_dir / "loss_curves.png",
            )
            figures.append(path)

        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> list[Path]:
    """Generate the post-training figures for the REINFORCE agents.

    Produces:
    - Reward curve (raw + moving average)
    - Loss curves (if available in step_losses)

    Parameters
    ----------
    history:
        History dictionary containing 'episode_rewards' and possibly
        'step_losses'.
    config:
        Experiment configuration (approach, env_id).
    output_dir:
        Root directory of the run — the figures are in {output_dir}/figures/.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    return ReinforceReporting().generate_figures(history, config, output_dir)
