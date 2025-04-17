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
    ):
        state = randn(self.state_shape, device = self.device)
        return state

    def forward(
        self,
        actions,
    ):
        state = randn(self.state_shape, device = self.device)
        reward = randint(0, 5, (), device = self.device).float()
        done = zeros((), device = self.device, dtype = torch.bool)

        return state, reward, done
