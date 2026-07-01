"""Monte-Carlo Tree Search for pedagogical MuZero."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class MinMaxStats:
    """Track finite Q ranges for MuZero min-max normalization."""

    minimum: float = field(default=float("inf"))
    maximum: float = field(default=float("-inf"))

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        if self.maximum <= self.minimum:
            return value
        return (value - self.minimum) / (self.maximum - self.minimum)


@dataclass
class Node:
    """One node in the latent MuZero search tree."""

    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    hidden_state: torch.Tensor | None = None
    children: dict[int, "Node"] = field(default_factory=dict)

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def policy(self) -> np.ndarray:
        if not self.children:
            return np.empty(0, dtype=np.float32)
        max_action = max(self.children) + 1
        visits = np.zeros(max_action, dtype=np.float32)
        for action, child in self.children.items():
            visits[action] = child.visit_count
        total = float(visits.sum())
        if total <= 0.0:
            return np.full_like(visits, 1.0 / max(1, len(visits)))
        return visits / total


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats, config: Any) -> float:
    pb_c_base = float(_cfg_value(config, "pb_c_base", 19_652.0))
    pb_c_init = float(_cfg_value(config, "pb_c_init", 1.25))
    pb_c = math.log((parent.visit_count + pb_c_base + 1.0) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(max(parent.visit_count, 1)) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count == 0:
        value_score = 0.0
    else:
        # Two-player (negamax): the child is the opponent's node, so its value is
        # negated when viewed from the parent. The backup stores each node's value
        # in its own perspective, so selection must flip the child's sign here.
        two_player = bool(_cfg_value(config, "two_player", False))
        child_value = -child.value() if two_player else child.value()
        q_value = child.reward + float(_cfg_value(config, "discount", 0.997)) * child_value
        value_score = min_max_stats.normalize(q_value)
    return value_score + prior_score


def select_child(parent: Node, min_max_stats: MinMaxStats, config: Any) -> tuple[int, Node]:
    candidates = sorted(parent.children.items(), key=lambda item: item[0])
    return max(
        candidates,
        key=lambda item: ucb_score(parent, item[1], min_max_stats, config),
    )


def expand_node(
    node: Node,
    *,
    policy_logits: torch.Tensor,
    value: float,
    legal_actions: np.ndarray | None = None,
    to_play: int | None = None,
) -> float:
    priors = torch.softmax(policy_logits.reshape(-1), dim=0).detach().cpu().numpy()
    if legal_actions is not None:
        mask = np.asarray(legal_actions, dtype=np.float32)
        priors = priors * mask
    total = float(priors.sum())
    if total <= 0.0:
        if legal_actions is None:
            priors = np.full_like(priors, 1.0 / max(1, len(priors)))
        else:
            legal = np.flatnonzero(np.asarray(legal_actions))
            priors = np.zeros_like(priors)
            priors[legal] = 1.0 / max(1, len(legal))
    else:
        priors = priors / total
    child_to_play = node.to_play if to_play is None else to_play
    node.children = {
        action: Node(prior=float(prior), to_play=child_to_play)
        for action, prior in enumerate(priors)
        if prior > 0.0
    }
    return float(value)


def add_exploration_noise(
    node: Node,
    *,
    dirichlet_alpha: float,
    exploration_fraction: float,
    rng: np.random.Generator,
) -> None:
    if not node.children:
        return
    actions = sorted(node.children)
    noise = rng.dirichlet([dirichlet_alpha] * len(actions))
    for action, sample in zip(actions, noise):
        child = node.children[action]
        child.prior = (1.0 - exploration_fraction) * child.prior + exploration_fraction * float(sample)


def backpropagate(
    search_path: list[Node],
    *,
    value: float,
    discount: float,
    min_max_stats: MinMaxStats,
    two_player: bool,
) -> None:
    current = float(value)
    for node in reversed(search_path):
        node.value_sum += current
        node.visit_count += 1
        min_max_stats.update(node.value())
        if two_player:
            current = node.reward - discount * current
        else:
            current = node.reward + discount * current


def run_mcts(
    root: Node,
    model_fns: dict[str, Any],
    config: Any,
    legal_actions: np.ndarray | None = None,
    to_play: int = 1,
    rng: np.random.Generator | None = None,
    add_root_noise: bool = True,
) -> Node:
    if root.hidden_state is None:
        raise ValueError("root.hidden_state must be set before running MCTS.")

    prediction = model_fns["prediction"]
    dynamics = model_fns["dynamics"]
    min_max_stats = MinMaxStats()
    root.to_play = to_play

    if not root.expanded():
        policy_logits, root_value = prediction(root.hidden_state)
        expand_node(
            root,
            policy_logits=policy_logits.squeeze(0),
            value=float(root_value.reshape(-1)[0].item()),
            legal_actions=legal_actions,
            to_play=to_play if not _cfg_value(config, "two_player", False) else -to_play,
        )
    if add_root_noise:
        add_exploration_noise(
            root,
            dirichlet_alpha=float(_cfg_value(config, "dirichlet_alpha", 0.25)),
            exploration_fraction=float(_cfg_value(config, "exploration_fraction", 0.25)),
            rng=rng or np.random.default_rng(int(_cfg_value(config, "seed", 0))),
        )

    for _ in range(int(_cfg_value(config, "num_simulations", 25))):
        node = root
        search_path = [node]
        actions: list[int] = []

        while node.expanded():
            action, node = select_child(search_path[-1], min_max_stats, config)
            search_path.append(node)
            actions.append(action)
            if not node.expanded():
                break

        parent = search_path[-2] if len(search_path) > 1 else None
        leaf = search_path[-1]
        if parent is not None and leaf.hidden_state is None:
            action_tensor = torch.tensor([actions[-1]], device=parent.hidden_state.device)
            hidden_state, reward = dynamics(parent.hidden_state, action_tensor)
            leaf.hidden_state = hidden_state
            leaf.reward = float(reward.reshape(-1)[0].item())
            leaf.to_play = parent.to_play if not _cfg_value(config, "two_player", False) else -parent.to_play

        if leaf.hidden_state is None:
            raise RuntimeError("Leaf node hidden state was not populated.")

        policy_logits, value = prediction(leaf.hidden_state)
        expand_node(
            leaf,
            policy_logits=policy_logits.squeeze(0),
            value=float(value.reshape(-1)[0].item()),
            legal_actions=None,
            to_play=leaf.to_play if not _cfg_value(config, "two_player", False) else -leaf.to_play,
        )
        backpropagate(
            search_path,
            value=float(value.reshape(-1)[0].item()),
            discount=float(_cfg_value(config, "discount", 0.997)),
            min_max_stats=min_max_stats,
            two_player=bool(_cfg_value(config, "two_player", False)),
        )

    return root
