"""PETS — Probabilistic Ensembles with Trajectory Sampling (Chua et al. 2018).

Importing this package triggers registration of the 'pets' config and agent
factory in the global registries, enabling the CLI to discover and launch
experiments from a YAML config.
"""

# Import order matters: config must register before agent/training import it.
from rl_from_scratch.pets.config import PetsConfig  # noqa: F401
from rl_from_scratch.pets.agent import PetsAgent  # noqa: F401
from rl_from_scratch.pets.training import train_pets  # noqa: F401

__all__ = ["PetsAgent", "PetsConfig", "train_pets"]
