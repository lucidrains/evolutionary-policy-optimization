import torch
from torch.nn import Module

from evolutionary_policy_optimization.epo import create_agent, Agent

class GymnasiumEnvWrapper(Module):
    def __init__(
        self,
        env
    ):
        super().__init__()
        self.env = env

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def to_agent_hparams(self):
        return dict(
            dim_state = self.env.observation_space.shape[0],
            actor_num_actions = self.env.action_space.n
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
