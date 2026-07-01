"""Actor-Critic methods (A2C, A2C-GAE, and A3C)."""

from rl_from_scratch.actor_critic.agent import A2CAgent, A2CGAEAgent, A3CAgent
from rl_from_scratch.actor_critic.buffer import RolloutBuffer
from rl_from_scratch.actor_critic.config import A2CConfig, A2CGAEConfig, A3CConfig
from rl_from_scratch.actor_critic.network import GaussianActor, CriticNetwork
from rl_from_scratch.actor_critic.optim import SharedAdam
from rl_from_scratch.actor_critic.training import train_a2c, train_a2c_gae, train_a3c

__all__ = [
    "GaussianActor",
    "CriticNetwork",
    "RolloutBuffer",
    "SharedAdam",
    "A2CAgent",
    "A2CGAEAgent",
    "A3CAgent",
    "A2CConfig",
    "A2CGAEConfig",
    "A3CConfig",
    "train_a2c",
    "train_a2c_gae",
    "train_a3c",
]
