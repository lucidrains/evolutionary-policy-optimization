import pytest

import torch
from evolutionary_policy_optimization import (
    LatentGenePool,
    Actor,
    Critic
)

@pytest.mark.parametrize('latent_ids', (2, (2, 4)))
@pytest.mark.parametrize('num_islands', (1, 4))
@pytest.mark.parametrize('sampled_mutation_strengths', (False, True))
def test_readme(
    latent_ids,
    num_islands,
    sampled_mutation_strengths
):

    latent_pool = LatentGenePool(
        num_latents = 128,
        dim_latent = 32,        
        num_islands = num_islands,
        fast_genetic_algorithm = sampled_mutation_strengths
    )

    state = torch.randn(2, 512)

    actor = Actor(dim_state = 512, dim_hiddens = (256, 128), num_actions = 4, dim_latent = 32)
    critic = Critic(dim_state = 512, dim_hiddens = (256, 128, 64), dim_latent = 32)

    latent = latent_pool(latent_id = latent_ids, state = state)

    actions = actor(state, latent)
    value = critic(state, latent)

    # interact with environment and receive rewards, termination etc

    # derive a fitness score for each gene / latent

    fitness = torch.randn(128)

    latent_pool.genetic_algorithm_step(fitness, migrate = num_islands > 1) # update once

    latent_pool.firefly_step(fitness)

@pytest.mark.parametrize('latent_ids', (2, (2, 4)))
def test_create_agent(
    latent_ids
):
    from evolutionary_policy_optimization import create_agent

    agent = create_agent(
        dim_state = 512,
        num_latents = 128,
        dim_latent = 32,
        actor_num_actions = 5,
        actor_dim_hiddens = (256, 128),
        critic_dim_hiddens = (256, 128, 64)
    )

    state = torch.randn(2, 512)

    actions = agent.get_actor_actions(state, latent_id = latent_ids)
    value = agent.get_critic_values(state, latent_id = latent_ids)

    # interact with environment and receive rewards, termination etc

    # derive a fitness score for each gene / latent

    fitness = torch.randn(128)

    agent.update_latent_gene_pool_(fitness) # update once

    # saving and loading

    agent.save('./agent.pt', overwrite = True)
    agent.load('./agent.pt')

@pytest.mark.parametrize('frozen_latents', (False, True))
@pytest.mark.parametrize('use_critic_ema', (False, True))
@pytest.mark.parametrize('critic_use_regression', (False, True))
@pytest.mark.parametrize('use_improved_critic_loss', (False, True))
@pytest.mark.parametrize('num_latents', (1, 8))
@pytest.mark.parametrize('diversity_aux_loss_weight', (0., 1e-3))
def test_e2e_with_mock_env(
    frozen_latents,
    use_critic_ema,
    num_latents,
    diversity_aux_loss_weight,
    critic_use_regression,
    use_improved_critic_loss
):
    from evolutionary_policy_optimization import create_agent, EPO, Env

    agent = create_agent(
        dim_state = 512,
        num_latents = num_latents,
        dim_latent = 32,
        actor_num_actions = 5,
        actor_dim_hiddens = (256, 128),
        critic_dim_hiddens = (256, 128, 64),
        use_critic_ema = use_critic_ema,
        diversity_aux_loss_weight = diversity_aux_loss_weight,
        critic_kwargs = dict(
            use_regression = critic_use_regression
        ),
        use_improved_critic_loss = use_improved_critic_loss,
        latent_gene_pool_kwargs = dict(
            frozen_latents = frozen_latents,
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False
    )

    epo = EPO(
        agent,
        episodes_per_latent = 1,
        max_episode_length = 10,
        action_sample_temperature = 1.
    )

    env = Env((512,))

    epo(agent, env, num_learning_cycles = 2)

    # saving and loading

    agent.save('./agent.pt', overwrite = True)
    agent.load('./agent.pt')
