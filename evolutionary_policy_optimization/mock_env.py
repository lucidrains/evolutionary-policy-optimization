from __future__ import annotations

from random import choice

import numpy as np
import torch
from torch import randint, randn, tensor
from torch.nn import Module

# helpers

def cast_tuple(v):
    return v if isinstance(v, tuple) else (v,)

def rng_or_none(seed):
    return None if seed is None else int(seed)

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
            truncated = tensor(choice((True, False)), device = self.device)
            terminated = tensor(choice((True, False)), device = self.device)
        else:
            truncated = terminated = tensor(False, device = self.device)

        self._step.add_(1)

        out = (state, reward, terminated, truncated)
        return (*tuple(t.numpy() for t in out), None)

# vectorized mock env - `num_envs` independent workers in one object, where
# each worker's trajectory is deterministic given its seed. satisfies the
# vectorized interface used by EPO for parallel rollout (`num_envs`,
# `reset_one`, `step_batch`), stepping only the listed workers

class VectorEnv(Module):
    def __init__(
        self,
        state_shape: int | tuple[int, ...],
        num_envs = 4,
        can_terminate_after = 2
    ):
        super().__init__()
        self.state_shape = cast_tuple(state_shape)
        self.num_envs = num_envs
        self.can_terminate_after = can_terminate_after

        # per-worker rng and step count, derived from the worker index

        self._worker_steps = np.zeros(num_envs, dtype = np.int64)
        self._worker_rngs = [np.random.default_rng(i) for i in range(num_envs)]

    def _next_transition(self, i):
        rng = self._worker_rngs[i]
        state = rng.standard_normal(self.state_shape).astype(np.float32)
        reward = float(rng.integers(0, 5))

        if self._worker_steps[i] > self.can_terminate_after:
            terminated = bool(rng.choice((True, False)))
            truncated = bool(rng.choice((True, False)))
        else:
            terminated = truncated = False

        self._worker_steps[i] += 1
        return state, reward, terminated, truncated

    def reset_one(
        self,
        i,
        seed = None
    ):
        self._worker_rngs[i] = np.random.default_rng(rng_or_none(seed))
        self._worker_steps[i] = 0

        state = self._worker_rngs[i].standard_normal(self.state_shape).astype(np.float32)
        return state, None

    def step_batch(
        self,
        actions,
        worker_ids = None
    ):
        worker_ids = np.arange(self.num_envs) if worker_ids is None else np.asarray(worker_ids)
        actions = np.asarray(actions)

        states = np.zeros((len(worker_ids),) + self.state_shape, dtype = np.float32)
        rewards = np.zeros(len(worker_ids), dtype = np.float32)
        terminated = np.zeros(len(worker_ids), dtype = bool)
        truncated = np.zeros(len(worker_ids), dtype = bool)

        for k, i in enumerate(worker_ids):
            states[k], rewards[k], terminated[k], truncated[k] = self._next_transition(int(i))

        return states, rewards, terminated, truncated

    def close(self):
        pass
