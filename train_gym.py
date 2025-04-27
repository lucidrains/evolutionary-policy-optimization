import torch

from evolutionary_policy_optimization import (
    EPO,
    GymnasiumEnvWrapper
)

# gymnasium

import gymnasium as gym

env = gym.make(
    'LunarLander-v3',
    render_mode = 'rgb_array'
)

env = GymnasiumEnvWrapper(env)

# epo

agent = env.to_epo_agent(
    num_latents = 8,
    dim_latent = 32,
    actor_dim_hiddens = (256, 128),
    critic_dim_hiddens = (256, 128, 64),
    latent_gene_pool_kwargs = dict(
        frac_natural_selected = 0.5,
        frac_tournaments = 0.5
    ),
    accelerate_kwargs = dict(
        cpu = False
    )
)

epo = EPO(
    agent,
    episodes_per_latent = 5,
    max_episode_length = 10,
    action_sample_temperature = 1.,
)

epo(agent, env, num_learning_cycles = 5)

agent.save('./agent.pt', overwrite = True)
