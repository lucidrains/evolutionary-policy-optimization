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
from shutil import rmtree

import gymnasium as gym
from accelerate import Accelerator

from evolutionary_policy_optimization import (
    EPO,
    GymnasiumEnvWrapper
)

def train(
    cpu = False,
    num_learning_cycles = 1000,
    target_reward = 50.,
    num_episodes_for_target = 20,
    use_wandb = False
):
    accelerator_kwargs = dict(cpu = cpu)

    if use_wandb:
        accelerator_kwargs.update(log_with = 'wandb')

    accelerator = Accelerator(**accelerator_kwargs)

    if use_wandb:
        accelerator.init_trackers('epo-lunar-lander')

    env = gym.make(
        'LunarLander-v3',
        render_mode = 'rgb_array'
    )

    if accelerator.is_main_process:
        rmtree('./recordings', ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = './recordings',
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: (eps_num % 250) == 0,
            disable_logger = True
        )

    env = GymnasiumEnvWrapper(env)

    # agent

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
        use_state_norm = False,
        accelerator = accelerator,
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

    # train

    from tqdm import tqdm

    recent_rewards = deque(maxlen = num_episodes_for_target)

    pbar = tqdm(range(num_learning_cycles), desc = 'learning cycle', disable = not accelerator.is_main_process)

    for cycle in pbar:
        memories_and_rewards = epo.gather_experience_from(env)
        agent.learn_from(memories_and_rewards)

        rewards = memories_and_rewards.cumulative_rewards
        fitness_var = rewards.mean(dim = -1).var().item()

        for r in rewards.flatten().tolist():
            recent_rewards.append(r)

        avg_reward = np.mean(recent_rewards)

        pbar.set_postfix(
            avg_reward = f"{avg_reward:.2f}",
            fitness_var = f"{fitness_var:.2f}"
        )

        if use_wandb:
            accelerator.log(dict(avg_reward = avg_reward, fitness_var = fitness_var), step = cycle)

        if len(recent_rewards) == num_episodes_for_target and avg_reward >= target_reward:
            accelerator.print(f'\ntarget reward of {target_reward} reached!')
            break

    if accelerator.is_main_process:
        agent.save('./agent.pt', overwrite = True)

    if use_wandb:
        accelerator.end_training()

if __name__ == '__main__':
    fire.Fire(train)
