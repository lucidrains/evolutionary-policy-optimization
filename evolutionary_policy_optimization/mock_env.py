from __future__ import annotations
from random import choice

import torch
from torch import tensor, randn, randint
from torch.nn import Module

# functions

def cast_tuple(v):
    return v if isinstance(v, tuple) else (v,)

# mock env

class Env(Module):
    def __init__(
        self,
        state_shape: int | tuple[int, ...],
        can_terminate_after = 2
    ):
        super().__init__()
        self.state_shape = cast_tuple(state_shape)

        self.can_terminate_after = can_terminate_after
        self.register_buffer('_step', tensor(0))

    @property
    def device(self):
        return self._step.device

    def reset(
        self,
        seed = None
    ):
        state = randn(self.state_shape, device = self.device)
        self._step.zero_()
        return state.numpy(), None

    def step(
        self,
        actions,
    ):
        state = randn(self.state_shape, device = self.device)
        reward = randint(0, 5, (), device = self.device).float()

        if self._step > self.can_terminate_after:
            truncated = tensor(choice((True, False)), device =self.device)
            terminated = tensor(choice((True, False)), device =self.device)
        else:
            truncated = terminated = tensor(False, device = self.device)

        self._step.add_(1)

        out = (state, reward, truncated, terminated)
        return (*tuple(t.numpy() for t in out), None)
