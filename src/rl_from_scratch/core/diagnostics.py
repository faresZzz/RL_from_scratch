"""Shared action diagnostics for continuous-control agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ActionDiagnosticsMixin:
    """Accumulate per-update action statistics (magnitude and clip fraction).

    The owning agent is responsible for accumulating ``_action_abs_sum``,
    ``_action_clip_fraction_sum`` and ``_action_metric_count`` while it acts
    (the accumulation step is action-representation specific). This mixin only
    provides the reset and consume helpers, which are identical across agents.
    """

    _action_abs_sum: float
    _action_clip_fraction_sum: float
    _action_metric_count: int

    def _reset_action_diagnostics(self) -> None:
        self._action_abs_sum = 0.0
        self._action_clip_fraction_sum = 0.0
        self._action_metric_count = 0

    def _consume_action_diagnostics(
        self, *, default_actions: torch.Tensor
    ) -> dict[str, float]:
        if self._action_metric_count:
            action_abs_mean = self._action_abs_sum / self._action_metric_count
            action_clip_fraction = (
                self._action_clip_fraction_sum / self._action_metric_count
            )
        else:
            action_abs_mean = float(default_actions.abs().mean().item())
            action_clip_fraction = 0.0
        self._reset_action_diagnostics()
        return {
            "action_abs_mean": float(action_abs_mean),
            "action_clip_fraction": float(action_clip_fraction),
        }
