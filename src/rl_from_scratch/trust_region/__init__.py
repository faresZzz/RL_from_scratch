"""Trust-region methods (TRPO and PPO)."""

from rl_from_scratch.trust_region.agent import TRPOAgent, PPOAgent
from rl_from_scratch.trust_region.config import TRPOConfig, PPOConfig
from rl_from_scratch.trust_region.training import train_trpo, train_ppo

__all__ = [
    "TRPOAgent",
    "PPOAgent",
    "TRPOConfig",
    "PPOConfig",
    "train_trpo",
    "train_ppo",
]
