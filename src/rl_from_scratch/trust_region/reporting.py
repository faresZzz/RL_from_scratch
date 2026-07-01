"""Post-training report for trust-region agents (TRPO, PPO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_from_scratch.core.reporting import (
    BaseReporting,
    plot_generic_metric,
)


class TrustRegionReporting(BaseReporting):
    """Post-training figures for TRPO and PPO agents.

    In addition to the base learning curve, produces:
    - Policy/value loss curves (if available in step_losses).
    - KL divergence curve (if available).
    - Entropy curve (if available, PPO only).
    """

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the figures for trust-region agents.

        Parameters
        ----------
        history:
            Dictionary containing ``episode_rewards`` and optionally
            ``step_losses`` with the keys ``policy_loss``, ``value_loss``,
            ``kl``, ``entropy``.
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

        approach = getattr(config, "approach", "trust_region")

        loss_series: dict[str, list[float]] = {}
        policy_losses = history.get("step_policy_losses", [])
        value_losses = history.get("step_value_losses", [])
        kl_series = history.get("step_kl", [])
        entropy_series = history.get("step_entropies", [])

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

        if kl_series:
            max_kl = getattr(config, "max_kl", None) or getattr(config, "target_kl", None)
            path = self.plot_kl_divergence(
                kl_series,
                title=f"KL divergence — {approach}",
                output_path=fig_dir / "kl_divergence.png",
                max_kl=max_kl,
            )
            figures.append(path)

        if entropy_series:
            path = self.plot_entropy(
                entropy_series,
                title=f"Policy entropy — {approach}",
                output_path=fig_dir / "entropy.png",
            )
            figures.append(path)

        optional_metrics = [
            (
                history.get("step_explained_variances", []),
                "Explained variance",
                "Explained variance — {approach}",
                "explained_variance.png",
            ),
            (
                history.get("step_grad_norms", []),
                "Gradient norm",
                "Gradient norm — {approach}",
                "grad_norm.png",
            ),
            (
                history.get("step_log_std_means", []),
                "log_std",
                "Policy log_std — {approach}",
                "log_std.png",
            ),
            (
                history.get("step_action_clip_fractions", []),
                "Action clipping",
                "Clipped env action fraction — {approach}",
                "action_clipping.png",
            ),
            (
                history.get("step_clip_fractions", []),
                "PPO clip fraction",
                "Clipped PPO ratio fraction — {approach}",
                "ppo_clip_fraction.png",
            ),
            (
                history.get("step_ratio_means", []),
                "PPO ratio mean",
                "Mean PPO ratio — {approach}",
                "ppo_ratio_mean.png",
            ),
            (
                history.get("step_ratio_stds", []),
                "PPO ratio std",
                "PPO ratio std — {approach}",
                "ppo_ratio_std.png",
            ),
            (
                history.get("step_line_search_accepts", []),
                "TRPO line search accept",
                "TRPO line search acceptance — {approach}",
                "trpo_line_search_accept.png",
            ),
            (
                history.get("step_line_search_step_fractions", []),
                "TRPO step fraction",
                "Accepted TRPO step fraction — {approach}",
                "trpo_step_fraction.png",
            ),
        ]
        for values, ylabel, title_template, filename in optional_metrics:
            if values:
                path = plot_generic_metric(
                    values,
                    name=ylabel,
                    title=title_template.format(approach=approach),
                    output_path=fig_dir / filename,
                )
                figures.append(path)

        return figures


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> list[Path]:
    """Generate the post-training figures for trust-region agents.

    Produces:
    - Reward curve (raw + moving average)
    - Policy/value loss curves (if available in step_losses)
    - KL divergence curve (if available in step_losses)
    - Entropy curve (if available in step_losses, PPO only)

    Parameters
    ----------
    history:
        History dictionary containing 'episode_rewards' and optionally
        'step_losses' with the keys 'policy_loss', 'value_loss', 'kl', 'entropy'.
    config:
        Experiment configuration (approach, env_id).
    output_dir:
        Root directory of the run — figures are in {output_dir}/figures/.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    return TrustRegionReporting().generate_figures(history, config, output_dir)
