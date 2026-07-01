"""Multi-seed benchmark for reproducible comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# Side-effect import: triggers auto-discovery of all sub-packages,
# which registers all @register_config/@register_agent decorators.
import rl_from_scratch  # noqa: F401


def run_benchmark(config_path: str, seeds: list[int] | None = None) -> dict[str, Any]:
    """Run training for each seed and aggregate the results.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    seeds:
        List of seeds to run.

    Returns
    -------
    dict
        Aggregated summary with cross-seed statistics.
    """
    from rl_from_scratch.core.config import AGENT_FACTORIES, load_config

    config = load_config(config_path)
    approach = config.approach
    resolved_seeds = list(seeds) if seeds is not None else list(range(config.num_seeds))

    if approach not in AGENT_FACTORIES:
        raise ValueError(
            f"Approach '{approach}' not registered. "
            f"Available: {list(AGENT_FACTORIES.keys())}"
        )
    if not resolved_seeds:
        raise ValueError("Benchmark requires at least one seed.")

    train_fn = AGENT_FACTORIES[approach]
    base_run_name = config.run_name or approach
    base_output_dir = config.output_dir or "runs"

    all_results = []

    for seed in resolved_seeds:
        run_name = f"{base_run_name}-seed{seed}"
        print(f"\n{'='*60}")
        print(f"  Benchmark: {approach} | seed={seed} | run={run_name}")
        print(f"{'='*60}\n")

        result = train_fn(
            config,
            output_dir=base_output_dir,
            run_name=run_name,
            seed=seed,
        )
        entry = {
            "seed": seed,
            "run_name": run_name,
            "result": result,
        }
        _require_deterministic_eval(entry)
        all_results.append(entry)

    # Aggregate results
    summary = _aggregate_results(all_results, approach, config_path, config)

    # Save summary
    summary_dir = _make_unique_summary_dir(base_output_dir, base_run_name)
    summary_path = summary_dir / "summary.json"
    summary["summary_dir"] = str(summary_dir)
    summary["summary_path"] = str(summary_path)

    save_summary = _make_json_serializable(summary)
    with open(summary_path, "w") as f:
        json.dump(save_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Benchmark complete: {approach}")
    print(f"  Seeds: {resolved_seeds}")
    print(f"  Summary saved to: {summary_path}")
    _print_summary(summary)
    print(f"{'='*60}\n")

    return summary


def _aggregate_results(
    all_results: list[dict[str, Any]],
    approach: str,
    config_path: str,
    config: Any | None = None,
) -> dict[str, Any]:
    """Aggregate per-seed results into cross-seed statistics."""
    if config is None:
        config = SimpleNamespace(
            env_id=None,
            total_timesteps=None,
            eval_every=None,
            eval_episodes=None,
            eval_seed=None,
            seed=None,
            metadata={},
        )
    run_dirs: list[str] = []
    seeds = [entry["seed"] for entry in all_results]

    for entry in all_results:
        paths = entry["result"].get("paths")
        if paths is not None:
            run_dir = getattr(paths, "run_dir", None)
            if run_dir is None and isinstance(paths, dict):
                run_dir = paths.get("run_dir", "")
            run_dirs.append(str(run_dir) if run_dir is not None else "")

    summary: dict[str, Any] = {
        "approach": approach,
        "config_path": str(config_path),
        "seeds": seeds,
        "num_seeds": len(all_results),
        "protocol": _build_protocol(config, all_results),
        "run_dirs": run_dirs,
    }

    metric_extractors: list[tuple[str, Any]] = [
        ("best_eval_mean_reward", _extract_best_eval_mean_reward),
        ("final_eval_mean_reward", _extract_final_eval_mean_reward),
        ("final_eval_timestep", _extract_final_eval_timestep),
        ("final_action_clip_fractions", lambda entry: _extract_history_last(entry, "step_action_clip_fractions")),
        ("final_log_std_means", lambda entry: _extract_history_last(entry, "step_log_std_means")),
        ("final_explained_variances", lambda entry: _extract_history_last(entry, "step_explained_variances")),
        ("final_action_abs_means", lambda entry: _extract_history_last(entry, "step_action_abs_means")),
        ("final_noise_stds", lambda entry: _extract_history_last(entry, "step_noise_stds")),
        ("final_q_means", lambda entry: _extract_history_last(entry, "step_q_means")),
        ("final_q1_means", lambda entry: _extract_history_last(entry, "step_q1_means")),
        ("final_q2_means", lambda entry: _extract_history_last(entry, "step_q2_means")),
        ("final_q_gaps", lambda entry: _extract_history_last(entry, "step_q_gaps")),
        ("final_target_q_means", lambda entry: _extract_history_last(entry, "step_target_q_means")),
        ("final_actor_update_flags", lambda entry: _extract_history_last(entry, "step_actor_update_flags")),
        ("final_alphas", lambda entry: _extract_history_last(entry, "step_alphas")),
        ("final_entropies", lambda entry: _extract_history_last(entry, "step_entropies")),
        ("final_alpha_losses", lambda entry: _extract_history_last(entry, "step_alpha_losses")),
        ("final_log_prob_means", lambda entry: _extract_history_last(entry, "step_log_prob_means")),
    ]

    for metric_name, extractor in metric_extractors:
        metric = _summarize_metric(all_results, extractor)
        if metric is not None:
            summary[metric_name] = metric

    mean_last_20 = _summarize_metric(all_results, _extract_mean_reward_last_20)
    if mean_last_20 is not None:
        summary["mean_reward_last_20"] = mean_last_20

    return summary


def _make_unique_summary_dir(base_output_dir: str, base_run_name: str) -> Path:
    """Create a unique summary directory so a previous benchmark is never overwritten."""
    root = Path(base_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{base_run_name}-benchmark-{timestamp}"
    candidate = root / base_name
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base_name}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _require_deterministic_eval(entry: dict[str, Any]) -> None:
    """Reject a seed that does not provide a deterministic evaluation trace."""
    result = entry.get("result", {})
    history = result.get("history", {}) or {}
    metrics = result.get("metrics", {}) or {}
    seed = entry.get("seed")

    explicit_flags = [
        history.get("eval_deterministic"),
        metrics.get("eval_deterministic"),
    ]
    if any(flag is False for flag in explicit_flags):
        raise ValueError(
            f"Seed {seed} refuses benchmark aggregation: deterministic evaluation is explicitly false."
        )

    has_eval_series = bool(history.get("eval_mean_rewards")) or ("final_eval_mean_reward" in metrics)
    if not has_eval_series:
        raise ValueError(
            f"Seed {seed} refuses benchmark aggregation: missing deterministic evaluation results."
        )


def _build_protocol(config: Any, all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the benchmark protocol block for the aggregated summary."""
    eval_seed = getattr(config, "eval_seed", None)
    if eval_seed is None:
        metadata = getattr(config, "metadata", {}) or {}
        eval_seed = metadata.get("eval_seed", getattr(config, "seed", None))

    cadence_steps, cadence_source = _infer_eval_cadence_steps(all_results, config)

    return {
        "env_id": getattr(config, "env_id", None),
        "total_timesteps": getattr(config, "total_timesteps", None),
        "eval_cadence_steps": cadence_steps,
        "eval_cadence_source": cadence_source,
        "legacy_eval_every_episodes": (
            getattr(config, "eval_every", None) if cadence_steps is None else None
        ),
        "eval_episodes": getattr(config, "eval_episodes", None),
        "eval_seed": eval_seed,
        "deterministic_eval_required": True,
    }


def _infer_eval_cadence_steps(
    all_results: list[dict[str, Any]],
    config: Any,
) -> tuple[int | None, str]:
    """Infer the evaluation cadence in timesteps when the information is available."""
    config_eval_every_steps = getattr(config, "eval_every_steps", None)
    if config_eval_every_steps is not None:
        return int(config_eval_every_steps), "config.eval_every_steps"

    for entry in all_results:
        history = entry["result"].get("history", {}) or {}
        eval_timesteps = history.get("eval_timesteps") or []
        if len(eval_timesteps) >= 2:
            deltas = [int(curr - prev) for prev, curr in zip(eval_timesteps, eval_timesteps[1:])]
            if deltas:
                return max(deltas), "history.eval_timesteps"
        if len(eval_timesteps) == 1:
            return int(eval_timesteps[0]), "history.eval_timesteps"

    return None, "legacy_episode_cadence"


def _extract_history_last(entry: dict[str, Any], key: str) -> float | None:
    """Return the last value of a history series if available."""
    history = entry["result"].get("history", {}) or {}
    values = history.get(key) or []
    if not values:
        return None
    return float(values[-1])


def _extract_best_eval_mean_reward(entry: dict[str, Any]) -> float | None:
    history = entry["result"].get("history", {}) or {}
    metrics = entry["result"].get("metrics", {}) or {}
    eval_means = history.get("eval_mean_rewards") or []
    if eval_means:
        return float(max(eval_means))
    value = metrics.get("best_eval_mean_reward")
    return float(value) if value is not None else None


def _extract_final_eval_mean_reward(entry: dict[str, Any]) -> float | None:
    history = entry["result"].get("history", {}) or {}
    metrics = entry["result"].get("metrics", {}) or {}
    eval_means = history.get("eval_mean_rewards") or []
    if eval_means:
        return float(eval_means[-1])
    value = metrics.get("final_eval_mean_reward")
    return float(value) if value is not None else None


def _extract_final_eval_timestep(entry: dict[str, Any]) -> float | None:
    history = entry["result"].get("history", {}) or {}
    metrics = entry["result"].get("metrics", {}) or {}
    eval_timesteps = history.get("eval_timesteps") or []
    if eval_timesteps:
        return float(eval_timesteps[-1])
    value = metrics.get("final_eval_timestep")
    if value is not None:
        return float(value)
    return None


def _extract_mean_reward_last_20(entry: dict[str, Any]) -> float | None:
    history = entry["result"].get("history", {}) or {}
    metrics = entry["result"].get("metrics", {}) or {}
    episode_rewards = history.get("episode_rewards") or []
    if episode_rewards:
        return float(np.mean(episode_rewards[-20:]))
    value = metrics.get("mean_reward_last_20")
    return float(value) if value is not None else None


def _summarize_metric(
    all_results: list[dict[str, Any]],
    extractor: Any,
) -> dict[str, Any] | None:
    """Aggregate a numeric metric with traceability of contributing seeds."""
    values: list[float] = []
    contributing_seeds: list[int] = []
    for entry in all_results:
        value = extractor(entry)
        if value is None:
            continue
        values.append(float(value))
        contributing_seeds.append(int(entry["seed"]))

    if not values:
        return None

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "per_seed": values,
        "contributing_seeds": contributing_seeds,
        "contributing_seed_count": len(contributing_seeds),
    }


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print(f"\n  Approach: {summary['approach']}")
    print(f"  Seeds: {summary['seeds']}")
    protocol = summary.get("protocol", {})
    if protocol:
        print("  Protocol:")
        print(f"    Env: {protocol.get('env_id')}")
        print(f"    Total timesteps: {protocol.get('total_timesteps')}")
        print(
            f"    Eval cadence steps: {protocol.get('eval_cadence_steps')} "
            f"({protocol.get('eval_cadence_source')})"
        )
        print(f"    Eval episodes: {protocol.get('eval_episodes')}")
        print(f"    Eval seed: {protocol.get('eval_seed')}")

    for key in ["best_eval_mean_reward", "final_eval_mean_reward", "final_eval_timestep"]:
        if key in summary:
            data = summary[key]
            label = key.replace("_", " ").title()
            print(
                f"  {label}: {data['mean']:.1f} ± {data['std']:.1f} "
                f"(n={data['contributing_seed_count']})"
            )
            print(f"    Per seed: {[f'{v:.1f}' for v in data['per_seed']]}")

    for key in [
        "final_action_abs_means",
        "final_action_clip_fractions",
        "final_noise_stds",
        "final_q_means",
        "final_q1_means",
        "final_q2_means",
        "final_q_gaps",
        "final_target_q_means",
        "final_actor_update_flags",
        "final_alphas",
        "final_entropies",
        "final_alpha_losses",
        "final_log_prob_means",
        "final_log_std_means",
        "final_explained_variances",
    ]:
        if key in summary:
            data = summary[key]
            label = key.replace("_", " ").title()
            print(
                f"  {label}: {data['mean']:.4f} ± {data['std']:.4f} "
                f"(n={data['contributing_seed_count']})"
            )

    if "mean_reward_last_20" in summary:
        data = summary["mean_reward_last_20"]
        print(
            f"  Mean Reward Last 20: {data['mean']:.1f} ± {data['std']:.1f} "
            f"(secondary, n={data['contributing_seed_count']})"
        )


def main() -> None:
    """Main entry point for the multi-seed benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Run multi-seed benchmark for RL algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to run (default: range(config.num_seeds))",
    )

    args = parser.parse_args()
    run_benchmark(args.config, args.seeds)


if __name__ == "__main__":
    main()
