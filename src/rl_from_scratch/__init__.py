"""Fundamental reinforcement learning algorithms from scratch."""

import importlib
import pkgutil

from rl_from_scratch.core.base import BaseAgent, BaseConfig
from rl_from_scratch.core.config import load_config, register_agent, register_config

# Auto-discovery of all algorithm sub-packages to trigger
# the @register_config/@register_agent decorators on package import.
for _finder, _name, _ispkg in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    if _ispkg:
        importlib.import_module(_name)

__all__ = [
    "BaseAgent",
    "BaseConfig",
    "load_config",
    "register_agent",
    "register_config",
]
