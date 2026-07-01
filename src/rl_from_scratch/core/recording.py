"""Run-level history collection and persistence.

``RunRecorder`` collects raw episode / eval / update metrics into flat
lists.  ``RunManager`` wraps a recorder and adds experiment plumbing:
checkpoints, periodic evaluation, and end-of-run finalization (persist
history, generate figures, record a greedy video).

Training loops interact exclusively with ``RunManager`` — typically via
``from_config`` (factory), ``record_episode``, ``record_updates``,
``maybe_eval``, ``maybe_checkpoint``, and ``finalize_run``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rl_from_scratch.core.artifacts import (
    ExperimentPaths,
    create_experiment_run,
    save_checkpoint,
    save_json,
)
from rl_from_scratch.core.metrics import append_update_metrics, summarize_history
from rl_from_scratch.core.schedules import should_eval_episode

if TYPE_CHECKING:
    from rl_from_scratch.core.base import BaseAgent

logger = logging.getLogger("rl_from_scratch")


def record_policy_video(
    env_id: str,
    policy_fn: Callable[[Any], Any],
    *,
    video_dir: str | Path,
    episodes: int = 1,
    seed: int = 0,
    max_steps: int | None = None,
    name_prefix: str = "demo",
    env_kwargs: dict[str, Any] | None = None,
) -> list[Path]:
    """Record a deterministic policy with ``rgb_array`` rendering.

    This is the lowest-level video helper: callers provide a pure
    ``policy_fn(obs) -> action`` and receive the generated ``.mp4`` paths.
    Higher-level training loops usually call
    ``core.reporting.record_greedy_episode`` because it also knows how to
    normalize observations and call ``agent.select_action(...,
    deterministic=True)``.
    """
    import gymnasium as gym

    from rl_from_scratch.core.env import clip_action

    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array", **(env_kwargs or {}))
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=lambda episode_id: episode_id < int(episodes),
        name_prefix=name_prefix,
    )

    try:
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + episode)
            steps = 0
            while max_steps is None or steps < int(max_steps):
                action = clip_action(policy_fn(obs), env)
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
    finally:
        env.close()

    return sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)


def _default_history() -> dict[str, list[Any]]:
    return {
        "episode_rewards": [],
        "episode_lengths": [],
        "eval_steps": [],
        "eval_timesteps": [],
        "eval_mean_rewards": [],
        "eval_std_rewards": [],
        "eval_min_rewards": [],
        "eval_max_rewards": [],
        "eval_success_rates": [],
    }


@dataclass
class RunRecorder:
    """Collect flat public run history and derive the summary contract."""

    history: dict[str, list[Any]] = field(default_factory=_default_history)

    def record_episode(self, *, reward: float, length: int, **metrics: float) -> None:
        self.history["episode_rewards"].append(float(reward))
        self.history["episode_lengths"].append(int(length))
        for name, value in metrics.items():
            self.history.setdefault(f"{name}s", []).append(float(value))

    def record_evaluation(
        self,
        *,
        step: int,
        mean_reward: float,
        std_reward: float,
        min_reward: float,
        max_reward: float,
        success_rate: float | None = None,
        timestep: int | None = None,
    ) -> None:
        self.history["eval_steps"].append(int(step))
        if timestep is not None:
            self.history["eval_timesteps"].append(int(timestep))
        self.history["eval_mean_rewards"].append(float(mean_reward))
        self.history["eval_std_rewards"].append(float(std_reward))
        self.history["eval_min_rewards"].append(float(min_reward))
        self.history["eval_max_rewards"].append(float(max_reward))
        if success_rate is not None:
            self.history["eval_success_rates"].append(float(success_rate))

    def record_update(self, metrics: dict[str, float]) -> None:
        append_update_metrics(self.history, metrics)

    def finalize(
        self,
        *,
        total_timesteps: int | None = None,
        observed_timesteps: int | None = None,
        episodes_to_solve: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        metrics = summarize_history(
            self.history,
            total_timesteps=total_timesteps,
            observed_timesteps=observed_timesteps,
            episodes_to_solve=episodes_to_solve,
        )
        return {"history": self.history, "metrics": metrics}

    def persist(
        self,
        paths: ExperimentPaths,
        *,
        total_timesteps: int | None = None,
        observed_timesteps: int | None = None,
        episodes_to_solve: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        result = self.finalize(
            total_timesteps=total_timesteps,
            observed_timesteps=observed_timesteps,
            episodes_to_solve=episodes_to_solve,
        )
        save_json(result["history"], paths.history_path)
        save_json(result["metrics"], paths.metrics_path)
        return result


@dataclass
class RunManager:
    """Coordinate run plumbing without owning any algorithm logic.

    Typical usage in a training loop::

        manager = RunManager.from_config(config, agent=agent)
        for episode in range(...):
            ...  # collect + learn (algorithm-specific)
            manager.record_episode(reward=..., length=...)
            manager.record_updates(metrics)
            manager.maybe_checkpoint(step=episode)
            manager.maybe_eval(agent, episode=episode, timestep=global_step)
        return manager.finalize_run(agent, reporting_module=my_reporting)
    """

    paths: ExperimentPaths
    agent: BaseAgent | None = None
    recorder: RunRecorder = field(default_factory=RunRecorder)
    checkpoint_keep_last: int = 3
    keep_best_checkpoint: bool = True
    _best_eval_reward: float = float("-inf")
    best_eval_step: int | None = None
    best_eval_timestep: int | None = None
    checkpoint_paths: list[Path] = field(default_factory=list)
    _best_checkpoint_reward: dict[str, float] = field(
        default_factory=lambda: {"value": float("-inf")}
    )

    # --- Eval / checkpoint scheduling state (set by from_config) ----------
    _eval_every_episodes: int | None = None
    _eval_every_steps: int | None = None
    _eval_episodes: int = 5
    _eval_seed: int = 10_000
    _eval_env_id: str = ""
    _max_steps_per_episode: int = 1000
    _solved_reward: float | None = None
    _checkpoint_every: int = 50
    _next_checkpoint_step: int = 0
    _next_eval_step: int | None = None
    _episodes_to_solve: int | None = None
    _total_timesteps: int | None = None
    _evaluate_fn: Callable[..., dict[str, float]] | None = None
    _config: Any = None

    @property
    def history(self) -> dict[str, list[Any]]:
        return self.recorder.history

    @property
    def best_eval_reward(self) -> float | None:
        if self._best_eval_reward == float("-inf"):
            return None
        return self._best_eval_reward

    def record_episode(self, *, reward: float, length: int, **metrics: float) -> None:
        self.recorder.record_episode(reward=reward, length=length, **metrics)

    def record_updates(
        self,
        metrics: dict[str, float] | None = None,
        **named_metrics: float,
    ) -> None:
        update_metrics = dict(metrics or {})
        update_metrics.update(named_metrics)
        if update_metrics:
            self.recorder.record_update(update_metrics)

    def record_eval(
        self,
        *,
        step: int,
        mean_reward: float,
        std_reward: float,
        min_reward: float,
        max_reward: float,
        success_rate: float | None = None,
        timestep: int | None = None,
    ) -> bool:
        self.recorder.record_evaluation(
            step=step,
            mean_reward=mean_reward,
            std_reward=std_reward,
            min_reward=min_reward,
            max_reward=max_reward,
            success_rate=success_rate,
            timestep=timestep,
        )
        if float(mean_reward) <= self._best_eval_reward:
            return False
        self._best_eval_reward = float(mean_reward)
        self.best_eval_step = int(step)
        self.best_eval_timestep = int(timestep) if timestep is not None else None
        return True

    def checkpoint(
        self,
        *,
        step: int,
        agent: BaseAgent | None = None,
        current_reward: float | None = None,
        keep_best: bool | None = None,
    ) -> Path:
        checkpoint_agent = agent or self.agent
        if checkpoint_agent is None:
            raise ValueError("RunManager.checkpoint requires an agent.")
        checkpoint_path = save_checkpoint(
            checkpoint_agent,
            self.paths.checkpoint_dir,
            step=step,
            keep_last=self.checkpoint_keep_last,
            keep_best=self.keep_best_checkpoint if keep_best is None else keep_best,
            current_reward=current_reward,
            _best_reward=self._best_checkpoint_reward,
        )
        self.checkpoint_paths.append(checkpoint_path)
        return checkpoint_path

    def finish(
        self,
        *,
        total_timesteps: int | None = None,
        observed_timesteps: int | None = None,
        episodes_to_solve: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self.recorder.persist(
            self.paths,
            total_timesteps=total_timesteps,
            observed_timesteps=observed_timesteps,
            episodes_to_solve=episodes_to_solve,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        agent: BaseAgent | None = None,
        evaluate_fn: Callable[..., dict[str, float]] | None = None,
    ) -> RunManager:
        """Create a fully-configured manager from a config dataclass.

        Parameters
        ----------
        config:
            Any ``BaseConfig`` subclass.  The factory reads ``output_dir``,
            ``approach``, ``run_name``, ``checkpoint_keep_last``,
            ``checkpoint_every``, ``eval_every``, ``eval_every_steps``,
            ``eval_episodes``, ``eval_seed``, ``env_id``,
            ``max_steps_per_episode``, ``solved_reward``, and
            ``total_timesteps``.
        agent:
            The agent to checkpoint.
        evaluate_fn:
            A callable ``(agent, env_id, *, n_episodes, seed, max_steps,
            solved_reward) -> dict`` used by ``maybe_eval``.  If *None*,
            the default ``core.evaluate.evaluate`` is used.
        """
        paths = create_experiment_run(
            config.output_dir,
            approach=config.approach,
            run_name=config.run_name,
            config=config.to_dict(),
        )
        mgr = cls(
            paths=paths,
            agent=agent,
            checkpoint_keep_last=config.checkpoint_keep_last,
        )
        # Eval schedule
        mgr._eval_every_episodes = getattr(config, "eval_every", None)
        mgr._eval_every_steps = getattr(config, "eval_every_steps", None)
        mgr._eval_episodes = getattr(config, "eval_episodes", 5)
        mgr._eval_seed = getattr(config, "eval_seed", 10_000)
        mgr._eval_env_id = config.env_id
        mgr._max_steps_per_episode = getattr(config, "max_steps_per_episode", 1000)
        mgr._solved_reward = getattr(config, "solved_reward", None)
        mgr._total_timesteps = getattr(config, "total_timesteps", None)
        # Checkpoint schedule
        mgr._checkpoint_every = config.checkpoint_every
        mgr._next_checkpoint_step = config.checkpoint_every
        # Eval timestep scheduling
        mgr._next_eval_step = mgr._eval_every_steps
        mgr._evaluate_fn = evaluate_fn
        mgr._config = config
        return mgr

    # ------------------------------------------------------------------
    # Periodic evaluation  (replaces 40-line ``maybe_run_eval`` closures)
    # ------------------------------------------------------------------

    def maybe_eval(
        self,
        agent: Any,
        *,
        episode: int,
        timestep: int = 0,
        force: bool = False,
    ) -> dict[str, float] | None:
        """Run a greedy evaluation if the schedule says so.

        Handles schedule check, evaluation, recording, best-checkpoint,
        solved detection, and logging — all the boilerplate that was
        previously copy-pasted across every ``training.py``.

        Returns the eval result dict when an evaluation was run, else None.
        """
        # --- Schedule check -----------------------------------------------
        if force:
            eval_timesteps = self.history.get("eval_timesteps", [])
            should_eval = not eval_timesteps or eval_timesteps[-1] != timestep
        elif self._next_eval_step is not None:
            should_eval = timestep >= self._next_eval_step
        else:
            should_eval = should_eval_episode(
                episode=episode,
                eval_every=self._eval_every_episodes,
            )

        if not should_eval:
            return None

        # --- Evaluate -----------------------------------------------------
        evaluate_fn = self._evaluate_fn
        if evaluate_fn is None:
            from rl_from_scratch.core.evaluate import evaluate as _default_eval
            evaluate_fn = _default_eval

        eval_result = evaluate_fn(
            agent,
            self._eval_env_id,
            n_episodes=self._eval_episodes,
            seed=self._eval_seed,
            max_steps=self._max_steps_per_episode,
            solved_reward=self._solved_reward,
        )

        # --- Record -------------------------------------------------------
        improved = self.record_eval(
            step=episode,
            timestep=timestep,
            mean_reward=eval_result["mean_reward"],
            std_reward=eval_result["std_reward"],
            min_reward=eval_result["min_reward"],
            max_reward=eval_result["max_reward"],
            success_rate=eval_result.get("success_rate"),
        )

        # Advance timestep-based eval schedule
        if self._next_eval_step is not None and self._eval_every_steps:
            while self._next_eval_step <= timestep:
                self._next_eval_step += self._eval_every_steps

        # Solved detection
        if (
            self._solved_reward is not None
            and eval_result["mean_reward"] >= self._solved_reward
            and self._episodes_to_solve is None
        ):
            self._episodes_to_solve = episode
            logger.info("Solved at episode %d!", episode)

        logger.info(
            "Eval (ep %d / step %d): mean=%.1f ± %.1f",
            episode,
            timestep,
            eval_result["mean_reward"],
            eval_result["std_reward"],
        )

        # Best checkpoint
        if improved:
            self.checkpoint(
                step=timestep or episode,
                keep_best=True,
                current_reward=eval_result["mean_reward"],
            )

        return eval_result

    # ------------------------------------------------------------------
    # Periodic checkpoint  (replaces 6-line blocks)
    # ------------------------------------------------------------------

    def maybe_checkpoint(self, *, step: int) -> Path | None:
        """Save a periodic (non-best) checkpoint if the schedule says so."""
        if step < self._next_checkpoint_step:
            return None
        path = self.checkpoint(step=step, keep_best=False)
        while self._next_checkpoint_step <= step:
            self._next_checkpoint_step += self._checkpoint_every
        return path

    # ------------------------------------------------------------------
    # End-of-run finalization  (replaces ~25-line blocks)
    # ------------------------------------------------------------------

    def finalize_run(
        self,
        agent: Any,
        *,
        reporting_module: Any = None,
        observed_timesteps: int | None = None,
        extra_metrics: dict[str, Any] | None = None,
        record_greedy_fn: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Persist history, generate figures, record greedy video, return result dict.

        Parameters
        ----------
        agent:
            The trained agent (used for greedy video and final checkpoint).
        reporting_module:
            Module with ``generate_training_figures(history, config, run_dir)``.
            Falls back to ``core.reporting.generate_training_figures``.
        observed_timesteps:
            Actual timesteps observed (for timestep-based loops).
        extra_metrics:
            Additional entries merged into the summary dict (e.g.
            ``final_epsilon``).
        record_greedy_fn:
            Override for the greedy video recorder (used by tabular which
            needs to pass a discretizer).  Defaults to
            ``core.reporting.record_greedy_episode``.
        """
        import rl_from_scratch.core.reporting as _core_reporting

        # Force a final eval if one hasn't run at this exact timestep
        ts = observed_timesteps or int(sum(self.history.get("episode_lengths", [])))
        ep = len(self.history.get("episode_rewards", []))
        self.maybe_eval(agent, episode=ep, timestep=ts, force=True)

        # Final checkpoint
        if self.history["episode_rewards"]:
            self.checkpoint(step=ts or ep, keep_best=False)

        # Persist history + compute summary
        persisted = self.finish(
            total_timesteps=self._total_timesteps,
            observed_timesteps=observed_timesteps
            or int(sum(self.history.get("episode_lengths", []))),
            episodes_to_solve=self._episodes_to_solve,
        )
        history = persisted["history"]
        metrics = persisted["metrics"]
        if extra_metrics:
            metrics.update(extra_metrics)

        # Figures
        reporting = reporting_module if reporting_module is not None else _core_reporting
        figures = reporting.generate_training_figures(history, self._config, self.paths.run_dir)

        # Greedy video — uses module-level lookup so tests can monkeypatch
        # ``rl_from_scratch.core.reporting.record_greedy_episode``.
        greedy_fn = record_greedy_fn or _core_reporting.record_greedy_episode
        video = greedy_fn(agent, self._config, self.paths.run_dir)

        for fig in figures:
            print(f"{fig}")
        if video:
            logger.info("Greedy video: %s", video)
            print(f"{video}")

        return {
            "agent": agent,
            "history": history,
            "metrics": metrics,
            "paths": self.paths,
        }
