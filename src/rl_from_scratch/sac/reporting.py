"""Post-training report for the SAC (Soft Actor-Critic) agent."""

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
    """Plot the evaluation diagnostics with best score and final score."""
    eval_means = _first_non_empty_series(history, "eval_mean_rewards", "eval_rewards")
    eval_stds = _first_non_empty_series(history, "eval_std_rewards")
    eval_mins = _first_non_empty_series(history, "eval_min_rewards")
    eval_maxs = _first_non_empty_series(history, "eval_max_rewards")

    if not eval_means:
        return output_path

    eval_steps, x_label = resolve_eval_axis(history, fallback_length=len(eval_means))

    n = min(len(eval_steps), len(eval_means))
    eval_steps = eval_steps[:n]
    eval_means = eval_means[:n]
    eval_stds = eval_stds[:n] if len(eval_stds) >= n else []
    eval_mins = eval_mins[:n] if len(eval_mins) >= n else []
    eval_maxs = eval_maxs[:n] if len(eval_maxs) >= n else []

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

    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


class SACReporting(BaseReporting):
    """Post-training figures for the SAC agent.

    Produces, in addition to the base learning curve:
    - Combined actor + critic loss curves.
    - Separate actor and critic loss curves.
    - Evolution of the entropy coefficient α.
    - Evolution of the entropy H(π).
    - Convergence of the α loss (auto-tuning).
    - Mean Q values (Q1, Q2) and Q_gap spread.
    - Evaluation diagnostics.
    - Mean log-probability of the policy.
    """

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the figures for the SAC agent.

        Parameters
        ----------
        history:
            Dictionary containing ``episode_rewards`` and possibly per-update
            metrics. Missing keys are ignored.
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

        approach = getattr(config, "approach", "sac")

        actor_losses = _first_non_empty_series(history, "step_actor_losses", "actor_losses")
        critic_losses = _first_non_empty_series(history, "step_critic_losses", "critic_losses")

        # Combined actor + critic loss curves
        loss_series: dict[str, list[float]] = {}
        if actor_losses:
            loss_series["actor_loss"] = actor_losses
        if critic_losses:
            loss_series["critic_loss"] = critic_losses

        if loss_series:
            path = self.plot_loss_curves(
                loss_series,
                title=f"Losses — {approach}",
                output_path=fig_dir / "loss_curves.png",
            )
            figures.append(path)

        # Separate loss curves
        for values, filename, ylabel in [
            (actor_losses, "actor_loss.png", "Actor loss"),
            (critic_losses, "critic_loss.png", "Critic loss"),
        ]:
            if values:
                path = plot_generic_metric(
                    values,
                    name=ylabel,
                    title=f"{ylabel} — {approach}",
                    output_path=fig_dir / filename,
                )
                figures.append(path)

        # Evaluation diagnostics
        eval_means = _first_non_empty_series(history, "eval_mean_rewards", "eval_rewards")
        if eval_means:
            path = _plot_eval_diagnostics(
                history,
                title=f"Evaluation diagnostics — {approach}",
                output_path=fig_dir / "eval_diagnostics.png",
            )
            figures.append(path)

        # SAC-specific metrics
        sac_metrics = [
            (
                ("step_alphas", "alphas"),
                "Alpha α",
                "Entropy coefficient α — {approach}",
                "alpha.png",
            ),
            (
                ("step_entropies", "entropies"),
                "Entropy H(π)",
                "Policy entropy H(π) — {approach}",
                "entropy.png",
            ),
            (
                ("step_alpha_losses", "alpha_losses"),
                "Alpha loss",
                "Convergence of the α loss — {approach}",
                "alpha_loss.png",
            ),
            (
                ("step_q1_means", "q1_means"),
                "Q1 mean",
                "Mean Q1 value — {approach}",
                "q_mean.png",
            ),
            (
                ("step_q_gaps", "q_gaps"),
                "Q gap |Q1 - Q2|",
                "Twin critics spread — {approach}",
                "q_gap.png",
            ),
            (
                ("step_target_q_means", "target_q_means"),
                "Target Q mean",
                "Mean target Q value — {approach}",
                "target_q_mean.png",
            ),
            (
                ("step_log_prob_means", "log_prob_means"),
                "Log prob mean",
                "Mean log-probability of the policy — {approach}",
                "log_prob.png",
            ),
        ]

        for keys, ylabel, title_template, filename in sac_metrics:
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
    """Generate the post-training figures for the SAC agent.

    Produces:
    - Reward curve (raw + moving average)
    - Combined and separate loss curves (actor, critic)
    - Evolution of α, H(π), α loss, Q-values, log-prob
    - Evaluation diagnostics if present

    Parameters
    ----------
    history:
        History dictionary containing 'episode_rewards' and possibly per-update
        metrics. Missing keys are ignored.
    config:
        Experiment configuration (approach, env_id).
    output_dir:
        Root directory of the run — the figures are in {output_dir}/figures/.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    return SACReporting().generate_figures(history, config, output_dir)
