"""Adam optimizer with state in shared memory for A3C.

Required so that torch.multiprocessing workers can share the first- and
second-moment statistics via POSIX shared memory.
"""

from __future__ import annotations

import torch
import torch.optim as optim


class SharedAdam(optim.Adam):
    """Adam with first- and second-moment state in shared memory.

    Allows A3C workers to read/write Adam's statistics from multiple
    processes via ``torch.multiprocessing`` without implicit copies.

    The state is initialized at construction time rather than at the first
    step, which is necessary in order to call ``share_memory_()`` before
    spawning the subprocesses.

    Parameters
    ----------
    params:
        Iterable of parameters or parameter groups.
    **kwargs:
        Any keyword arguments accepted by ``torch.optim.Adam``
        (``lr``, ``betas``, ``eps``, ``weight_decay``, ``amsgrad``).
    """

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)

        # Pre-initialize and move each parameter's state into shared memory.
        # Without this, Adam's state is only created on the first call to step(),
        # which prevents share_memory_() from being called before the spawn.
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1).share_memory_()
                state["exp_avg"] = torch.zeros_like(p.data).share_memory_()
                state["exp_avg_sq"] = torch.zeros_like(p.data).share_memory_()
