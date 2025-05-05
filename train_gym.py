import torch

from evolutionary_policy_optimization import (
    EPO,
    GymnasiumEnvWrapper
)

# gymnasium

from shutil import rmtree
import gymnasium as gym

env = gym.make(
    'LunarLander-v3',
    render_mode = 'rgb_array'
)

rmtree('./recordings', ignore_errors = True)

env = gym.wrappers.RecordVideo(
    env = env,
    video_folder = './recordings',
    name_prefix = 'lunar-video',
    episode_trigger = lambda eps_num: (eps_num % 250) == 0,
    disable_logger = True
)

env = GymnasiumEnvWrapper(env)

# epo

agent = env.to_epo_agent(
    num_latents = 8,
    dim_latent = 32,
    actor_dim = 128,
    actor_mlp_depth = 3,
    critic_dim = 256,
    critic_mlp_depth = 5,
    latent_gene_pool_kwargs = dict(
        frac_natural_selected = 0.5,
        frac_tournaments = 0.5
    ),
    accelerate_kwargs = dict(
        cpu = False
    ),
    actor_optim_kwargs = dict(
        cautious_factor = 0.1,
    ),
    critic_optim_kwargs = dict(
        cautious_factor = 0.1,
    ),
)

epo = EPO(
    agent,
    episodes_per_latent = 10,
    max_episode_length = 250,
    action_sample_temperature = 1.,
)

epo(agent, env, num_learning_cycles = 100)

agent.save('./agent.pt', overwrite = True)
