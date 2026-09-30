# -*- coding: utf-8 -*-
"""Epoch-level warmup + MultiStepLR scheduler.

Examples:
    In yaml::

        lr_scheduler:
          core_method: warmup_multistep
          gamma: 0.1
          step_size: [10, 25, 40]
          warmup_epochs: 2
          warmup_factor: 0.001
"""

from __future__ import annotations

from typing import Optional, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR


class WarmupMultiStepLR:
    """Linear per-iteration warmup followed by epoch-level MultiStep decay.

    The warmup phase ramps the lr from ``base_lr * warmup_factor`` to
    ``base_lr`` linearly over ``warmup_epochs`` epochs (measured in
    iterations so DDP runs ramp identically). After warmup the schedule
    behaves exactly like ``MultiStepLR``.

    Designed for the repo's train.py which calls ``scheduler.step(epoch)``
    once per epoch and ``train_utils.setup_lr_schedular`` which replays
    ``scheduler.step()`` (no-arg) when resuming.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Sequence[int],
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        warmup_factor: float = 1.0e-3,
        n_iter_per_epoch: Optional[int] = None,
    ) -> None:
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.warmup_factor = float(warmup_factor)
        self.n_iter_per_epoch = (
            int(n_iter_per_epoch) if n_iter_per_epoch else 0
        )
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._inner = MultiStepLR(
            optimizer, milestones=list(milestones), gamma=float(gamma)
        )
        self._last_iter = 0
        if self.warmup_epochs > 0:
            self._apply_warmup(self._last_iter)

    # ------------------------------------------------------------------ #
    # warmup helpers
    # ------------------------------------------------------------------ #
    def _total_warmup_iters(self) -> int:
        return self.warmup_epochs * max(1, self.n_iter_per_epoch)

    def _apply_warmup(self, iteration: int) -> None:
        total = self._total_warmup_iters()
        if total <= 0 or iteration >= total:
            return
        alpha = iteration / total
        scale = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
        for base_lr, group in zip(self._base_lrs, self._inner.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def _in_warmup(self) -> bool:
        total = self._total_warmup_iters()
        return total > 0 and self._last_iter < total

    def step_iter(self) -> None:
        """Advance one training iteration during warmup (no-op afterwards)."""
        if not self._in_warmup():
            return
        self._last_iter += 1
        self._apply_warmup(self._last_iter)

    # ------------------------------------------------------------------ #
    # scheduler-compatible interface
    # ------------------------------------------------------------------ #
    def step(self, epoch: Optional[int] = None) -> None:
        """Epoch-level step; mirrors MultiStepLR then re-imposes warmup.

        ``train.py`` calls this once at the end of ``epoch``. Re-applying
        the ramp only while iterations are still inside the warmup window
        leaves the full base lr in place once warmup finishes, so the
        following epoch does not stay one step short of ``base_lr``.
        """
        if epoch is None:
            self._inner.step()
        else:
            self._inner.step(epoch)
        if self._in_warmup():
            self._apply_warmup(self._last_iter)

    def get_last_lr(self) -> list:
        return self._inner.get_last_lr()

    def state_dict(self) -> dict:
        return {
            "inner": self._inner.state_dict(),
            "last_iter": self._last_iter,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._inner.load_state_dict(state_dict.get("inner", state_dict))
        self._last_iter = int(state_dict.get("last_iter", 0))
