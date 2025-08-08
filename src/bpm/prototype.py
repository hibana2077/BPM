from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class MomentumPrototype:
    tau: float = 0.99

    def __post_init__(self):
        self.buffer = None

    @torch.no_grad()
    def update(self, batch_mean: torch.Tensor):
        if self.buffer is None:
            self.buffer = batch_mean.detach().clone()
        else:
            self.buffer.mul_(self.tau).add_(batch_mean.detach(), alpha=1 - self.tau)
        return self.buffer


def batch_mean(images: torch.Tensor) -> torch.Tensor:
    """Compute per-batch mean image: (1, C, H, W)."""
    return images.mean(dim=0, keepdim=True)
