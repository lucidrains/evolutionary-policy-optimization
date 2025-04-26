# gymnasium

import gymnasium as gym

env = gym.make(
    'LunarLander-v3',
    render_mode = 'rgb_array'
)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# epo

import torch

from evolutionary_policy_optimization import (
    create_agent,
    EPO,
    Env
)

agent = create_agent(
    dim_state = state_dim,
    num_latents = 1,
    dim_latent = 32,
    actor_num_actions = num_actions,
    actor_dim_hiddens = (256, 128),
    critic_dim_hiddens = (256, 128, 64),
    latent_gene_pool_kwargs = dict(
        frac_natural_selected = 0.5
    )
)

epo = EPO(
    agent,
    episodes_per_latent = 1,
    max_episode_length = 10,
    action_sample_temperature = 1.
)

epo.to('cpu' if not torch.cuda.is_available() else 'cuda')

epo(agent, env, num_learning_cycles = 5)
