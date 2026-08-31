from math import prod

import torch
from torch.nn import Module

from evolutionary_policy_optimization.epo import Agent, create_agent, exists

def rescale_from_to(x, from_range = (0., 1.), to_range = (-1., 1.)):
    # e.g. beta actions on (0, 1) -> the env's action bounds

    from_low, from_high = from_range
    to_low, to_high = to_range

    if torch.is_tensor(x):
        dd_kwargs = dict(device = x.device, dtype = x.dtype)
        from_low, from_high, to_low, to_high = [torch.as_tensor(t, **dd_kwargs) for t in (from_low, from_high, to_low, to_high)]

    return to_low + (to_high - to_low) * (x - from_low) / (from_high - from_low)

class GymnasiumEnvWrapper(Module):
    def __init__(
        self,
        env,
        rescale_to = None # e.g. (-2., 2.) for pendulum's torque bounds
    ):
        super().__init__()
        self.env = env

        if not exists(rescale_to) and not hasattr(env.action_space, 'n'):
            rescale_to = (env.action_space.low, env.action_space.high)

        self.rescale_to = rescale_to

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        # beta lives on (0, 1) - rescale to the env's bounds at the interface

        if exists(self.rescale_to):
            actions = rescale_from_to(actions, to_range = self.rescale_to)

        return self.env.step(actions, *args, **kwargs)

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def to_agent_hparams(self):
        action_space = self.env.action_space
        is_continuous = not hasattr(action_space, 'n')

        num_actions = action_space.n if not is_continuous else prod(action_space.shape)

        return dict(
            dim_state = self.env.observation_space.shape[0],
            actor_num_actions = num_actions,
            action_is_continuous = is_continuous
        )

    def to_epo_agent(
        self,
        *args,
        **kwargs
    ) -> Agent:

        return create_agent(
            *args,
            **self.to_agent_hparams(),
            **kwargs
        )
