"""Shared visualization primitives for all algorithms.

Each function creates a figure, saves it as PNG (dpi=150, bbox_inches='tight'),
closes the figure to avoid memory leaks, and returns the path of the created file.

The ``BaseReporting`` class wraps the base functions and defines the
``generate_figures`` interface that each specific module can override.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rl_from_scratch.core.utils import moving_average

# Use a non-interactive backend for headless compatibility
matplotlib.use("Agg")

logger = logging.getLogger("rl_from_scratch")


def _ensure_figures_dir(output_dir: Path) -> Path:
    """Create and return the figures directory inside output_dir."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def resolve_eval_axis(
    history: dict[str, Any],
    *,
    fallback_length: int | None = None,
) -> tuple[list[int], str]:
    """Resolve the evaluation X axis, preferring timesteps.

    Returns ``(values, label)`` where ``label`` is ``"Timesteps"`` if
    ``eval_timesteps`` is available, otherwise ``"Episode"`` for ``eval_steps``.
    A 1..N index is used as a last-resort fallback for legacy histories.
    """
    eval_timesteps = history.get("eval_timesteps")
    if eval_timesteps is not None and len(eval_timesteps) > 0:
        return list(eval_timesteps), "Timesteps"

    eval_steps = history.get("eval_steps")
    if eval_steps is not None and len(eval_steps) > 0:
        return list(eval_steps), "Episode"

    if fallback_length is not None and fallback_length > 0:
        return list(range(1, fallback_length + 1)), "Episode"

    return [], "Episode"


def plot_reward_curve(
    rewards: list[float],
    title: str,
    output_path: Path,
    window: int = 50,
) -> Path:
    """Plot the raw reward curve + moving average.

    Parameters
    ----------
    rewards:
        List of per-episode rewards.
    title:
        Plot title.
    output_path:
        Full output path (PNG).
    window:
        Window for the moving average (default 50).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    episodes = list(range(1, len(rewards) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Raw reward")

    if len(rewards) >= window:
        ma = [
            float(np.mean(rewards[max(0, i - window):i + 1]))
            for i in range(len(rewards))
        ]
        ax.plot(episodes, ma, color="steelblue", linewidth=2, label=f"Moving average (w={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_curves(
    step_losses_dict: dict[str, list[float]],
    title: str,
    output_path: Path,
) -> Path:
    """Plot one or more loss curves over the course of updates.

    Parameters
    ----------
    step_losses_dict:
        Dictionary name → list of loss values.
    title:
        Plot title.
    output_path:
        Full output path (PNG).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, values in step_losses_dict.items():
        if values:
            ax.plot(values, alpha=0.7, label=name)

    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_epsilon_decay(
    epsilons: list[float],
    title: str,
    output_path: Path,
) -> Path:
    """Plot the decay of epsilon over the course of episodes.

    Parameters
    ----------
    epsilons:
        List of per-episode epsilon values.
    title:
        Plot title.
    output_path:
        Full output path (PNG).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    episodes = list(range(1, len(epsilons) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, epsilons, color="darkorange", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_kl_divergence(
    kl_values: list[float],
    title: str,
    output_path: Path,
    max_kl: float | None = None,
) -> Path:
    """Plot the KL divergence over the course of updates.

    Parameters
    ----------
    kl_values:
        List of KL values per update.
    title:
        Plot title.
    output_path:
        Full output path (PNG).
    max_kl:
        Maximum KL threshold to display as a reference line (optional).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kl_values, color="crimson", linewidth=1.5, label="KL divergence")

    if max_kl is not None:
        ax.axhline(max_kl, color="gray", linestyle="--", linewidth=1, label=f"max_kl={max_kl}")

    ax.set_xlabel("Update")
    ax.set_ylabel("KL divergence")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_entropy(
    entropies: list[float],
    title: str,
    output_path: Path,
) -> Path:
    """Plot the policy entropy over the course of updates.

    Parameters
    ----------
    entropies:
        List of entropy values per update.
    title:
        Plot title.
    output_path:
        Full output path (PNG).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(entropies, color="mediumseagreen", linewidth=1.5)
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_generic_metric(
    values: list[float],
    name: str,
    title: str,
    output_path: Path,
) -> Path:
    """Plot a generic metric over time.

    Parameters
    ----------
    values:
        List of values to plot.
    name:
        Name of the metric (used for the Y axis).
    title:
        Plot title.
    output_path:
        Full output path (PNG).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _maybe_normalize(agent: Any, obs: Any) -> Any:
    """Normalize the observation via agent.obs_normalizer if present (frozen stats)."""
    normalizer = getattr(agent, "obs_normalizer", None)
    if normalizer is not None:
        return normalizer.normalize(np.asarray(obs, dtype=np.float32), update=False)
    return obs


def _clip_action(action: Any, env: Any) -> Any:
    """Clip the action to the action space bounds if continuous."""
    if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
        return np.clip(action, env.action_space.low, env.action_space.high)
    return action


def record_greedy_episode(
    agent: Any,
    config: Any,
    output_dir: Path,
    *,
    discretizer: Any | None = None,
    video_dir: Path | None = None,
    render_mode: str = "rgb_array",
    episode_trigger: Callable[[int], bool] | None = None,
    step_trigger: Callable[[int], bool] | None = None,
    video_length: int = 0,
    name_prefix: str = "greedy-episode",
    fps: int | None = None,
    disable_logger: bool = True,
    env_kwargs: dict[str, Any] | None = None,
) -> Path | None:
    """Record a greedy episode as video (headless-compatible).

    Uses render_mode="rgb_array" + gymnasium.wrappers.RecordVideo by default
    to guarantee compatibility with headless environments (nohup, SSH).

    Parameters
    ----------
    agent:
        Trained agent implementing select_action(obs, deterministic=True).
    config:
        Configuration containing env_id, seed, max_steps_per_episode.
    output_dir:
        Run root directory — the video is saved in {output_dir}/figures/.
    discretizer:
        Optional discretizer for tabular agents.
    video_dir:
        Explicit video directory. If absent, the video is saved in a unique
        subdirectory under ``{output_dir}/figures/videos/``.  The unique
        folder avoids Gymnasium's overwrite warning when a run directory is
        reused during smoke tests.
    env_kwargs:
        Optional keyword arguments forwarded to ``gym.make``.  Use this when
        evaluation/recording must match non-default environment construction
        such as HalfCheetah with current positions included.
    render_mode:
        Gymnasium render mode used to create the environment. To record a
        video, use a capturable mode such as "rgb_array". "human" is refused
        because it displays live and does not provide frames.
    episode_trigger, step_trigger, video_length, name_prefix, fps, disable_logger:
        Options passed through to gymnasium.wrappers.RecordVideo.

    Returns
    -------
    Path | None
        Path of the created video file, or None on error.
    """
    import time

    import gymnasium as gym

    if render_mode == "human":
        raise ValueError(
            "record_greedy_episode requires a frame-producing render mode; "
            "render_mode='human' displays live and cannot be recorded with "
            "RecordVideo. Use render_mode='rgb_array' for recording or a "
            "separate live render helper."
        )

    if video_dir is None:
        target_dir = (
            _ensure_figures_dir(output_dir)
            / "videos"
            / f"{name_prefix}-{time.time_ns()}"
        )
    else:
        target_dir = video_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        env = gym.make(config.env_id, render_mode=render_mode, **(env_kwargs or {}))
        record_kwargs: dict[str, Any] = {
            "video_folder": str(target_dir),
            "video_length": video_length,
            "name_prefix": name_prefix,
            "disable_logger": disable_logger,
        }
        if episode_trigger is not None:
            record_kwargs["episode_trigger"] = episode_trigger
        else:
            record_kwargs["episode_trigger"] = lambda episode: episode == 0
        if step_trigger is not None:
            record_kwargs["step_trigger"] = step_trigger
        if fps is not None:
            record_kwargs["fps"] = fps

        env = gym.wrappers.RecordVideo(env, **record_kwargs)

        try:
            obs, _ = env.reset(seed=config.seed)
            if discretizer is not None:
                state = discretizer.transform(obs)
            else:
                state = _maybe_normalize(agent, obs)

            max_steps = getattr(config, "max_steps_per_episode", 1000)
            for _ in range(max_steps):
                action = agent.select_action(state, deterministic=True)
                action = _clip_action(action, env)
                obs, _, terminated, truncated, _ = env.step(action)
                if discretizer is not None:
                    state = discretizer.transform(obs)
                else:
                    state = _maybe_normalize(agent, obs)
                if terminated or truncated:
                    break
        finally:
            env.close()

        # Look for the most recent video created in the directory
        videos = sorted(target_dir.glob("*.mp4"))
        return videos[-1] if videos else None

    except Exception as exc:
        logger.warning("record_greedy_episode failed: %s: %s", type(exc).__name__, exc)
        return None


def plot_learning_curves(
    train_rewards: list[float],
    eval_steps: list[int] | None,
    eval_means: list[float] | None,
    eval_stds: list[float] | None,
    title: str,
    output_path: Path,
    window: int = 50,
    solved_reward: float | None = None,
    x_label: str = "Episode",
) -> Path:
    """Unified learning curve: raw training + moving average + eval.

    Plots the training rewards (raw with transparency + moving average)
    and, if available, the evaluation rewards with a standard-deviation band.
    An optional horizontal line indicates the solved threshold.

    Parameters
    ----------
    train_rewards:
        Per-episode training rewards.
    eval_steps:
        Episode indices at which evaluation took place (optional).
    eval_means:
        Mean evaluation rewards (optional).
    eval_stds:
        Standard deviations of the evaluation rewards (optional).
    title:
        Plot title.
    output_path:
        Full output path (PNG).
    window:
        Window for the moving average.
    solved_reward:
        Solved reward threshold to display (optional).
    x_label:
        Label of the X axis (``"Episode"`` or ``"Timesteps"`` depending on the source).

    Returns
    -------
    Path
        Path of the created PNG file.
    """
    episodes = list(range(1, len(train_rewards) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw rewards (transparent)
    ax.plot(episodes, train_rewards, alpha=0.2, color="steelblue", label="Raw reward")

    # Moving average
    if len(train_rewards) >= window:
        ma = [
            float(np.mean(train_rewards[max(0, i - window) : i + 1]))
            for i in range(len(train_rewards))
        ]
        ax.plot(
            episodes, ma, color="steelblue", linewidth=2,
            label=f"Moving average (w={window})"
        )

    # Evaluation curve with standard-deviation band
    if eval_steps and eval_means:
        ax.plot(eval_steps, eval_means, color="darkorange", linewidth=2,
                marker="o", markersize=4, label="Eval (mean)")
        if eval_stds:
            means_arr = np.array(eval_means)
            stds_arr = np.array(eval_stds)
            ax.fill_between(
                eval_steps,
                means_arr - stds_arr,
                means_arr + stds_arr,
                color="darkorange",
                alpha=0.2,
                label="Eval ± std",
            )

    # Solved threshold
    if solved_reward is not None:
        ax.axhline(
            solved_reward, color="green", linestyle="--", linewidth=1,
            label=f"Solved ({solved_reward})"
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


class BaseReporting:
    """Base class for post-training figure generation.

    The static methods expose the shared visualization primitives.
    The ``generate_figures`` method can be overridden by subclasses
    to add algorithm-specific figures.
    """

    # ------------------------------------------------------------------
    # Static methods — shared primitives
    # ------------------------------------------------------------------

    @staticmethod
    def plot_reward_curve(
        rewards: list[float],
        title: str,
        output_path: Path,
        window: int = 50,
    ) -> Path:
        """Plot the raw reward curve + moving average."""
        return plot_reward_curve(rewards, title, output_path, window=window)

    @staticmethod
    def plot_loss_curves(
        step_losses_dict: dict[str, list[float]],
        title: str,
        output_path: Path,
    ) -> Path:
        """Plot one or more loss curves."""
        return plot_loss_curves(step_losses_dict, title, output_path)

    @staticmethod
    def plot_epsilon_decay(
        epsilons: list[float],
        title: str,
        output_path: Path,
    ) -> Path:
        """Plot the decay of epsilon."""
        return plot_epsilon_decay(epsilons, title, output_path)

    @staticmethod
    def plot_kl_divergence(
        kl_values: list[float],
        title: str,
        output_path: Path,
        max_kl: float | None = None,
    ) -> Path:
        """Plot the KL divergence."""
        return plot_kl_divergence(kl_values, title, output_path, max_kl=max_kl)

    @staticmethod
    def plot_entropy(
        entropies: list[float],
        title: str,
        output_path: Path,
    ) -> Path:
        """Plot the policy entropy."""
        return plot_entropy(entropies, title, output_path)

    @staticmethod
    def plot_learning_curves(
        train_rewards: list[float],
        eval_steps: list[int] | None,
        eval_means: list[float] | None,
        eval_stds: list[float] | None,
        title: str,
        output_path: Path,
        solved_reward: float | None = None,
        x_label: str = "Episode",
    ) -> Path:
        """Unified learning curve with evaluation band."""
        return plot_learning_curves(
            train_rewards, eval_steps, eval_means, eval_stds,
            title, output_path, solved_reward=solved_reward, x_label=x_label,
        )

    @staticmethod
    def record_greedy_episode(
        agent: Any,
        config: Any,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path | None:
        """Record a greedy episode as video."""
        return record_greedy_episode(agent, config, output_dir, **kwargs)

    # ------------------------------------------------------------------
    # Main method — to be overridden by subclasses
    # ------------------------------------------------------------------

    def generate_figures(
        self,
        history: dict[str, Any],
        config: Any,
        output_dir: Path,
    ) -> list[Path]:
        """Generate the main learning curve.

        Produces a single ``learning_curves.png`` plot combining the
        training rewards and, if available, the evaluation metrics.
        Subclasses call ``super().generate_figures()`` and then add
        their own figures.

        Parameters
        ----------
        history:
            History dictionary (``episode_rewards``, ``eval_steps``, etc.).
        config:
            Experiment configuration (approach, env_id).
        output_dir:
            Run root directory.

        Returns
        -------
        list[Path]
            List of the created PNG paths.
        """
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        figures: list[Path] = []
        rewards = history.get("episode_rewards", [])

        if rewards:
            approach = getattr(config, "approach", "RL")
            env_id = getattr(config, "env_id", "")
            title = f"{approach} — {env_id}"
            # The combined curve keeps a consistent axis. Legacy histories
            # have no timestep axis for the training rewards, so we use
            # their episodic eval_steps rather than mixing units.
            eval_history = history
            if history.get("eval_timesteps") and not history.get("episode_timesteps"):
                eval_history = {**history, "eval_timesteps": []}
            eval_axis, eval_axis_label = resolve_eval_axis(
                eval_history,
                fallback_length=len(history.get("eval_mean_rewards", []) or []),
            )

            path = self.plot_learning_curves(
                rewards,
                eval_axis,
                history.get("eval_mean_rewards"),
                history.get("eval_std_rewards"),
                title,
                figures_dir / "learning_curves.png",
                x_label="Timesteps" if eval_axis_label == "Timesteps" else "Episode",
            )
            figures.append(path)

        return figures

    def _maybe_plot_metric(
        self,
        figures: list[Path],
        history: dict[str, Any],
        figures_dir: Path,
        *,
        key: str,
        name: str,
        title: str,
        filename: str,
    ) -> None:
        """Append a ``plot_generic_metric`` figure for ``history[key]`` when present.

        Centralises the ``series = history.get(key); if series: plot...`` guard
        that every algorithm reporting module repeats. Values are coerced to
        float; this is a no-op when the series is missing or empty.
        """
        series = history.get(key, [])
        if not series:
            return
        figures.append(
            plot_generic_metric(
                [float(v) for v in series],
                name=name,
                title=title,
                output_path=figures_dir / filename,
            )
        )


def generate_training_figures(
    history: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> list[Path]:
    """Generate the base figures: only the reward curve.

    Specific algorithm modules override this function to add their own
    figures (loss, epsilon, KL, entropy, etc.).

    Parameters
    ----------
    history:
        Training history dictionary ('episode_rewards' key expected).
    config:
        Experiment configuration (for the title).
    output_dir:
        Run root directory.

    Returns
    -------
    list[Path]
        List of the created PNG paths.
    """
    fig_dir = _ensure_figures_dir(output_dir)
    figures: list[Path] = []

    rewards = history.get("episode_rewards", [])
    if rewards:
        approach = getattr(config, "approach", "RL")
        env_id = getattr(config, "env_id", "")
        path = plot_reward_curve(
            rewards,
            title=f"Rewards — {approach} on {env_id}",
            output_path=fig_dir / "reward_curve.png",
        )
        figures.append(path)

    return figures
