"""Shared core infrastructure for RL from scratch."""

from rl_from_scratch.core.artifacts import ExperimentPaths, create_experiment_run, save_checkpoint, save_json
from rl_from_scratch.core.base import BaseAgent, BaseConfig
from rl_from_scratch.core.config import AGENT_FACTORIES, CONFIG_REGISTRY, build_agent, load_config, register_agent, register_config
from rl_from_scratch.core.diagnostics import ActionDiagnosticsMixin
from rl_from_scratch.core.env import ActionSpec, EnvSpec, ObservationSpec, clip_action, get_env_info, get_env_spec, make_env
from rl_from_scratch.core.metrics import append_update_metrics, history_key_for_update_metric, summarize_history
from rl_from_scratch.core.normalization import ObservationNormalizer, RunningMeanStd
from rl_from_scratch.core.recording import RunRecorder
from rl_from_scratch.core.schedules import every_n_episodes, every_n_steps, should_checkpoint_episode, should_checkpoint_timestep, should_eval_episode, should_eval_timestep, should_record_video
from rl_from_scratch.core.utils import moving_average, resolve_device, set_all_seeds, soft_update

__all__ = [
    "ActionDiagnosticsMixin",
    "ActionSpec",
    "AGENT_FACTORIES",
    "append_update_metrics",
    "BaseAgent",
    "BaseConfig",
    "build_agent",
    "clip_action",
    "CONFIG_REGISTRY",
    "create_experiment_run",
    "EnvSpec",
    "every_n_episodes",
    "every_n_steps",
    "ExperimentPaths",
    "get_env_info",
    "get_env_spec",
    "history_key_for_update_metric",
    "load_config",
    "make_env",
    "moving_average",
    "ObservationNormalizer",
    "ObservationSpec",
    "register_agent",
    "register_config",
    "resolve_device",
    "RunRecorder",
    "RunningMeanStd",
    "save_checkpoint",
    "save_json",
    "set_all_seeds",
    "should_checkpoint_episode",
    "should_checkpoint_timestep",
    "should_eval_episode",
    "should_eval_timestep",
    "should_record_video",
    "soft_update",
    "summarize_history",
]
