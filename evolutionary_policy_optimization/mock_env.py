from __future__ import annotations

import torch
from torch import tensor, randn, randint
from torch.nn import Module

# functions

def cast_tuple(v):
    return v if isinstance(v, tuple) else v\

# mock env

class Env(Module):
    def __init__(
        self,
        state_shape: int | tuple[int, ...]
    ):
        super().__init__()
        self.state_shape = cast_tuple(state_shape)
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
        done = torch.zeros((), device = self.device, dtype = torch.bool)

        return state, reward, done
