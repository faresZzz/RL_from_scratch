"""DreamerV1 — Dream to Control (Hafner et al. 2020).

Import order matters: config must be registered before agent/training
so that ``build_agent`` and ``load_config`` can find "dreamer".
"""

from rl_from_scratch.dreamer.config import DreamerConfig
from rl_from_scratch.dreamer.agent import DreamerAgent
from rl_from_scratch.dreamer.training import train_dreamer

__all__ = ["DreamerConfig", "DreamerAgent", "train_dreamer"]
