from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class EMA:
    decay: float = 0.99

    def __post_init__(self):
        self._shadow = {}
        self._initialized = False

    def _init(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.detach().clone()
        self._initialized = True

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if not self._initialized:
            self._init(model)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self._shadow
            self._shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self._shadow:
                param.data.copy_(self._shadow[name].data)
