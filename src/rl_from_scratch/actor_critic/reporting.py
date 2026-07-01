"""Post-training report for Actor-Critic agents (A2C, A2C-GAE, A3C)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rl_from_scratch.core.reporting import (
    BaseReporting,
    plot_generic_metric,
    resolve_eval_axis,
)


def _first_non_empty_series(history: dict[str, Any], *keys: str) -> list[float]:
    """Return the first non-empty series among several backward-compatible names."""
    for key in keys:
        values = history.get(key)
        if values is not None and len(values) > 0:
            return list(values)
    return []


def _plot_eval_diagnostics(
    history: dict[str, Any],
    title: str,
    output_path: Path,
) -> Path:
    """Plot the evaluation diagnostics, with best score and final score."""
    eval_means = _first_non_empty_series(history, "eval_mean_rewards", "eval_rewards")
    eval_stds = _first_non_empty_series(history, "eval_std_rewards")
    eval_mins = _first_non_empty_series(history, "eval_min_rewards")
    eval_maxs = _first_non_empty_series(history, "eval_max_rewards")
    success_rates = _first_non_empty_series(history, "eval_success_rates")

    if not eval_means:
        return output_path

    eval_steps, x_label = resolve_eval_axis(history, fallback_length=len(eval_means))

    n = min(len(eval_steps), len(eval_means))
    eval_steps = eval_steps[:n]
    eval_means = eval_means[:n]
    eval_stds = eval_stds[:n] if len(eval_stds) >= n else []
    eval_mins = eval_mins[:n] if len(eval_mins) >= n else []
    eval_maxs = eval_maxs[:n] if len(eval_maxs) >= n else []
    success_rates = success_rates[:n] if len(success_rates) >= n else []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_steps, eval_means, marker="o", linewidth=2, label="Eval mean")

    if eval_stds:
        means_arr = np.array(eval_means)
        stds_arr = np.array(eval_stds)
        ax.fill_between(
            eval_steps,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.2,
            label="Eval +/- std",
        )

    if eval_mins and eval_maxs:
        ax.plot(eval_steps, eval_mins, linestyle="--", linewidth=1, label="Eval min")
        ax.plot(eval_steps, eval_maxs, linestyle="--", linewidth=1, label="Eval max")

    best_idx = int(np.argmax(eval_means))
    final_idx = len(eval_means) - 1
    ax.scatter(
        [eval_steps[best_idx]],
        [eval_means[best_idx]],
        color="green",
        zorder=3,
        label=f"Best eval: {eval_means[best_idx]:.2f}",
    )
    ax.scatter(
        [eval_steps[final_idx]],
        [eval_means[final_idx]],
        color="crimson",
        zorder=3,
        label=f"Final eval: {eval_means[final_idx]:.2f}",
    )

    if success_rates:
        ax2 = ax.twinx()
        ax2.plot(
            eval_steps,
            success_rates,
            color="purple",
            alpha=0.6,
            linewidth=1.5,
            label="Success rate",
        )
        ax2.set_ylabel("Success rate")
        ax2.set_ylim(0.0, 1.0)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
    else:
        ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


class ActorCriticReporting(BaseReporting):
    """Post-training figures for Actor-Critic agents.

    In addition to the base learning curve, produces:
    - Policy/value loss curves (if available in step_losses).
    - Entropy curve (if available in step_losses).
    - Evaluation diagnostics, explained variance, gradient norms,
      log_std, and action clipping if these series exist in the history.
    """

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the figures for Actor-Critic agents.

        Parameters
        ----------
        history:
            Dictionary containing ``episode_rewards`` and optionally
            per-update metrics. Missing keys are ignored.
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

        approach = getattr(config, "approach", "a2c")

        loss_series: dict[str, list[float]] = {}
        policy_losses = _first_non_empty_series(history, "step_policy_losses", "policy_losses")
        value_losses = _first_non_empty_series(history, "step_value_losses", "value_losses")
        total_losses = _first_non_empty_series(
            history,
            "step_total_losses",
            "step_losses",
            "total_losses",
        )
        entropy_series = _first_non_empty_series(history, "step_entropies", "entropies")

        if policy_losses:
            loss_series["policy_loss"] = policy_losses
        if value_losses:
            loss_series["value_loss"] = value_losses
        if total_losses:
            loss_series["total_loss"] = total_losses

        if loss_series:
            path = self.plot_loss_curves(
                loss_series,
                title=f"Losses — {approach}",
                output_path=fig_dir / "loss_curves.png",
            )
            figures.append(path)

        for values, filename, ylabel in [
            (policy_losses, "policy_loss.png", "Policy loss"),
            (value_losses, "value_loss.png", "Value loss"),
            (total_losses, "total_loss.png", "Total loss"),
        ]:
            if values:
                path = plot_generic_metric(
                    values,
                    name=ylabel,
                    title=f"{ylabel} — {approach}",
                    output_path=fig_dir / filename,
                )
                figures.append(path)

        if entropy_series:
            path = self.plot_entropy(
                entropy_series,
                title=f"Policy entropy — {approach}",
                output_path=fig_dir / "entropy.png",
            )
            figures.append(path)

        eval_means = _first_non_empty_series(history, "eval_mean_rewards", "eval_rewards")
        if eval_means:
            path = _plot_eval_diagnostics(
                history,
                title=f"Evaluation diagnostics — {approach}",
                output_path=fig_dir / "eval_diagnostics.png",
            )
            figures.append(path)

        optional_metrics = [
            (
                (
                    "step_explained_variances",
                    "step_explained_variance",
                    "explained_variance",
                    "explained_variances",
                ),
                "Explained variance",
                "Explained variance — {approach}",
                "explained_variance.png",
            ),
            (
                ("step_grad_norms", "grad_norms", "grad_norm"),
                "Gradient norm",
                "Gradient norm — {approach}",
                "grad_norm.png",
            ),
            (
                (
                    "step_log_std_means",
                    "step_log_stds",
                    "log_stds",
                    "log_std",
                ),
                "log_std",
                "Policy log_std — {approach}",
                "log_std.png",
            ),
            (
                (
                    "step_action_clip_fractions",
                    "action_clip_fractions",
                    "action_clipping",
                    "clip_fractions",
                ),
                "Action clipping",
                "Clipped action fraction — {approach}",
                "action_clipping.png",
            ),
        ]
        for keys, ylabel, title_template, filename in optional_metrics:
            values = _first_non_empty_series(history, *keys)
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
    """Generate the post-training figures for Actor-Critic agents.

    Produces:
    - Reward curve (raw + moving average)
    - Combined and separate loss curves (if available)
    - Entropy curve (if available in step_losses)
    - Evaluation diagnostics and optional metrics if present

    Parameters
    ----------
    history:
        History dictionary containing 'episode_rewards' and optionally
        per-update metrics. Missing keys are ignored.
    config:
        Experiment configuration (approach, env_id).
    output_dir:
        Root directory of the run — figures are in {output_dir}/figures/.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    return ActorCriticReporting().generate_figures(history, config, output_dir)
