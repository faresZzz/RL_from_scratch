"""MBPO: Model-Based Policy Optimization (Janner et al. 2019)."""

from rl_from_scratch.mbpo.config import MbpoConfig
from rl_from_scratch.mbpo.agent import MbpoAgent
from rl_from_scratch.mbpo.training import train_mbpo

__all__ = ["MbpoAgent", "MbpoConfig", "train_mbpo"]
