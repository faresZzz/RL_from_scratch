"""Post-training report for Deep Q agents (DQN, Double DQN, Rainbow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import (
    BaseReporting,
)


class DQNReporting(BaseReporting):
    """Post-training figures for Deep Q agents.

    In addition to the base learning curve, produces:
    - Loss curves (if available in step_losses).
    - Epsilon decay curve.
    - Optional Rainbow diagnostics (mean Q, TD error, PER beta).
    """

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the figures for DQN agents.

        Parameters
        ----------
        history:
            Dictionary containing ``episode_rewards``, ``epsilons``,
            and optionally ``step_losses``.
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

        approach = getattr(config, "approach", "dqn")

        # Loss curves
        losses = history.get("step_losses", [])
        if losses:
            path = self.plot_loss_curves(
                {"loss": losses},
                title=f"Losses — {approach}",
                output_path=fig_dir / "loss_curves.png",
            )
            figures.append(path)

        # Epsilon decay
        epsilons = history.get("epsilons", [])
        if epsilons:
            path = self.plot_epsilon_decay(
                epsilons,
                title=f"Epsilon decay — {approach}",
                output_path=fig_dir / "epsilon_decay.png",
            )
            figures.append(path)

        q_means = history.get("step_q_means", [])
        td_errors = history.get("step_td_error_means", [])
        if q_means or td_errors:
            path = self.plot_loss_curves(
                {
                    "q_mean": q_means,
                    "td_error_mean": td_errors,
                },
                title=f"Diagnostics Q — {approach}",
                output_path=fig_dir / "q_diagnostics.png",
            )
            figures.append(path)

        betas = history.get("step_betas", [])
        if betas:
            path = self.plot_loss_curves(
                {"beta": betas},
                title=f"Annealing PER beta — {approach}",
                output_path=fig_dir / "per_beta.png",
            )
            figures.append(path)

        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> list[Path]:
    """Generate the post-training figures for Deep Q agents.

    Produces:
    - Reward curve (raw + moving average)
    - Loss curves (if available in step_losses)
    - Epsilon decay curve
    - Optional Rainbow diagnostics

    Parameters
    ----------
    history:
        History dictionary containing 'episode_rewards', 'epsilons',
        and optionally 'step_losses'.
    config:
        Experiment configuration (approach, env_id).
    output_dir:
        Root directory of the run — figures are in {output_dir}/figures/.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    return DQNReporting().generate_figures(history, config, output_dir)
