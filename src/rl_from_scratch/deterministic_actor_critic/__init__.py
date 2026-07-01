"""DDPG and TD3 subpackage — deterministic actor-critic algorithms.

Exposes the agents, configs, and training functions for deterministic-policy
algorithms (DDPG, TD3) for continuous action spaces.
"""

from __future__ import annotations

from rl_from_scratch.deterministic_actor_critic.agent import DDPGAgent, TD3Agent
from rl_from_scratch.deterministic_actor_critic.config import DDPGConfig, TD3Config
from rl_from_scratch.deterministic_actor_critic.training import train_ddpg, train_td3

__all__ = [
    "DDPGAgent",
    "TD3Agent",
    "DDPGConfig",
    "TD3Config",
    "train_ddpg",
    "train_td3",
]
