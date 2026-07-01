"""Post-training reporting for tabular agents (Q-learning, SARSA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import (
    BaseReporting,
)


class TabularReporting(BaseReporting):
    """Post-training figures for tabular agents.

    Produces, in addition to the base learning curve:
    - Epsilon decay curve.
    """

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the figures for tabular agents.

        Parameters
        ----------
        history:
            Dictionary containing ``episode_rewards`` and ``epsilons``.
        config:
            Experiment configuration.
        output_dir:
            Run root directory.

        Returns
        -------
        list[Path]
            Paths of the created PNGs.
        """
        figures = super().generate_figures(history, config, output_dir)
        fig_dir = output_dir / "figures"

        approach = getattr(config, "approach", "tabular")

        epsilons = history.get("epsilons", [])
        if epsilons:
            path = self.plot_epsilon_decay(
                epsilons,
                title=f"Epsilon decay — {approach}",
                output_path=fig_dir / "epsilon_decay.png",
            )
            figures.append(path)

        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> list[Path]:
    """Generate the post-training figures for tabular agents.

    Produces:
    - Reward curve (raw + moving average)
    - Epsilon decay curve

    Parameters
    ----------
    history:
        History dictionary containing 'episode_rewards' and 'epsilons'.
    config:
        Experiment configuration (approach, env_id).
    output_dir:
        Run root directory — figures are in {output_dir}/figures/.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    return TabularReporting().generate_figures(history, config, output_dir)
