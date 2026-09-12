# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "moviepy",
#     "numpy",
#     "swig",
#     "torch",
#     "wandb",
#     "evolutionary-policy-optimization",
# ]
# [tool.uv.sources]
# evolutionary-policy-optimization = { path = ".", editable = true }
# ///

import fire
import numpy as np
from collections import deque
from pathlib import Path
from shutil import rmtree

import gymnasium as gym
from accelerate import Accelerator

from evolutionary_policy_optimization import (
    EPO,
    GymnasiumEnvWrapper
)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

ENV_CONFIGS = dict(
    cartpole = dict(
        env_name = 'CartPole-v1',
        target_reward = 500.,
        max_episode_length = 500,
        episodes_per_latent = 1,
        actor_dim = 64,
        actor_mlp_depth = 2,
        critic_dim = 128,
        critic_mlp_depth = 3,
        hl_gauss_min = 0.,
        hl_gauss_max = 500.,
    ),
    pendulum = dict(
        env_name = 'Pendulum-v1',
        target_reward = -200.,
        max_episode_length = 200,
        episodes_per_latent = 2,
        actor_dim = 64,
        actor_mlp_depth = 2,
        critic_dim = 128,
        critic_mlp_depth = 3,
        hl_gauss_min = -2500.,
        hl_gauss_max = 0.,
        actor_kwargs = dict(
            beta_kwargs = dict(
                pos_fn = 'softplus',
                init_conc = 2.,
            )
        )
    ),
    inverted_pendulum = dict(
        env_name = 'InvertedPendulum-v5',
        target_reward = 500.,
        max_episode_length = 1000,
        episodes_per_latent = 2,
        actor_dim = 64,
        actor_mlp_depth = 2,
        critic_dim = 128,
        critic_mlp_depth = 3,
        hl_gauss_min = 0.,
        hl_gauss_max = 1000.,
    ),
    lunar = dict(
        env_name = 'LunarLander-v3',
        target_reward = 50.,
        max_episode_length = 250,
        episodes_per_latent = 2,
        actor_dim = 128,
        actor_mlp_depth = 3,
        critic_dim = 256,
        critic_mlp_depth = 5,
        hl_gauss_min = -200.,
        hl_gauss_max = 500.,
    ),
)

def train(
    cpu = False,
    env_name = 'lunar',
    num_learning_cycles = 1000,
    target_reward = None,
    num_episodes_for_target = 20,
    learning_epochs: int | None = None,
    use_wandb = False,
    resume = False
):
    assert env_name in ENV_CONFIGS, f'env_name must be one of {tuple(ENV_CONFIGS.keys())}'

    config = ENV_CONFIGS[env_name]

    learning_epochs = default(learning_epochs, config.get('learning_epochs', 2))

    accelerator_kwargs = dict(cpu = cpu)

    if use_wandb:
        accelerator_kwargs.update(log_with = 'wandb')

    accelerator = Accelerator(**accelerator_kwargs)

    if use_wandb:
        accelerator.init_trackers(f'epo-{env_name}')

    env = gym.make(
        config['env_name'],
        render_mode = 'rgb_array'
    )

    if accelerator.is_main_process:
        rmtree('./recordings', ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = './recordings',
            name_prefix = f'{env_name}-video',
            episode_trigger = lambda eps_num: (eps_num % 250) == 0,
            disable_logger = True
        )

    env = GymnasiumEnvWrapper(env)

    # agent

    agent_kwargs = dict(
        num_latents = 8,
        dim_latent = 32,
        actor_dim = config['actor_dim'],
        actor_mlp_depth = config['actor_mlp_depth'],
        critic_dim = config['critic_dim'],
        critic_mlp_depth = config['critic_mlp_depth'],
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.5,
            frac_tournaments = 0.5
        ),
        use_state_norm = False,
        accelerator = accelerator,
        critic_kwargs = dict(
            hl_gauss_loss_kwargs = dict(
                min_value = config['hl_gauss_min'],
                max_value = config['hl_gauss_max'],
                num_bins = 250,
            ),
        ),
        actor_kwargs = config.get('actor_kwargs', dict()),
    )

    agent_kwargs.update(config.get('agent_kwargs', dict()))

    agent = env.to_epo_agent(**agent_kwargs)

    checkpoint_path = f'./{env_name}_agent.pt'

    if resume and Path(checkpoint_path).exists():
        agent.load(checkpoint_path)

    epo = EPO(
        agent,
        episodes_per_latent = config['episodes_per_latent'],
        max_episode_length = config['max_episode_length'],
        action_sample_temperature = 1.,
    )

    # train

    from tqdm import tqdm

    target_reward = config['target_reward'] if target_reward is None else target_reward

    recent_rewards = deque(maxlen = num_episodes_for_target)

    pbar = tqdm(range(num_learning_cycles), desc = 'learning cycle', disable = not accelerator.is_main_process)

    for cycle in pbar:
        memories_and_rewards = epo.gather_experience_from(env)
        agent.learn_from(memories_and_rewards, epochs = learning_epochs)

        rewards = memories_and_rewards.cumulative_rewards
        fitness_var = rewards.mean(dim = -1).var().item()

        for r in rewards.flatten().tolist():
            recent_rewards.append(r)

        avg_reward = np.mean(recent_rewards)
        best_latent_reward = rewards.mean(dim = -1).max().item()

        pbar.set_postfix(
            avg_reward = f"{avg_reward:.2f}",
            best_reward = f"{best_latent_reward:.2f}",
            fitness_var = f"{fitness_var:.2f}"
        )

        if use_wandb:
            accelerator.log(dict(avg_reward = avg_reward, best_reward = best_latent_reward, fitness_var = fitness_var), step = cycle)

        if len(recent_rewards) == num_episodes_for_target and (avg_reward >= target_reward or best_latent_reward >= target_reward):
            accelerator.print(f'\ntarget reward of {target_reward} reached!')
            break

    if accelerator.is_main_process:
        agent.save(f'./{env_name}_agent.pt', overwrite = True)

    if use_wandb:
        accelerator.end_training()

if __name__ == '__main__':
    fire.Fire(train)
