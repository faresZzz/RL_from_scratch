"""PILCO and Deep PILCO — Probabilistic Inference for Learning COntrol.

Importing this package triggers registration of both 'pilco' and
'deep_pilco' configs and agent factories in the global registries,
enabling the CLI to discover and launch experiments from a YAML config.
"""

# Import order matters: configs must register before agents/training import them.
from rl_from_scratch.pilco.config import PilcoConfig, DeepPilcoConfig  # noqa: F401
from rl_from_scratch.pilco.agent import PilcoAgent, DeepPilcoAgent  # noqa: F401
from rl_from_scratch.pilco.training import train_pilco, train_deep_pilco  # noqa: F401

__all__ = [
    "PilcoAgent",
    "PilcoConfig",
    "train_pilco",
    "DeepPilcoAgent",
    "DeepPilcoConfig",
    "train_deep_pilco",
]
