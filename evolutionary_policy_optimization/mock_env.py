from __future__ import annotations

import torch
from torch import tensor, randn, randint
from torch.nn import Module

# mock env

class Env(Module):
    def __init__(
        self,
        state_shape: tuple[int, ...]
    ):
        super().__init__()
        self.state_shape = state_shape
        self.register_buffer('dummy', tensor(0))

    @property
    def device(self):
        return self.dummy.device

    def reset(
        self
    ) -> State:
        return randn(self.state_shape, device = self.device)

    def forward(
        self,
        actions: Int['a'],
    ) -> State:

        return randn(self.state_shape, device = self.device)
