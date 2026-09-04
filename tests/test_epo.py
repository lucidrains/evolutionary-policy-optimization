import pytest

import torch
from torch.distributions import Beta, Categorical

from evolutionary_policy_optimization.epo import (
    LatentGenePool,
    Actor,
    Critic,
    create_agent,
    shrink_and_perturb_,
    EPO
)

from evolutionary_policy_optimization.mock_env import Env, VectorEnv

@pytest.mark.parametrize('latent_ids', (2, (2, 4)))
@pytest.mark.parametrize('num_islands', (1, 4))
@pytest.mark.parametrize('sampled_mutation_strengths', (False, True))
@pytest.mark.parametrize('l2norm_latent', (False, True))
def test_readme(
    latent_ids,
    num_islands,
    sampled_mutation_strengths,
    l2norm_latent
):

    latent_pool = LatentGenePool(
        num_latents = 128,
        dim_latent = 32,
        l2norm_latent = l2norm_latent,
        num_islands = num_islands,
        fast_genetic_algorithm = sampled_mutation_strengths
    )

    state = torch.randn(2, 512)

    actor = Actor(dim_state = 512, dim = 256, mlp_depth = 2, num_actions = 4, dim_latent = 32)
    critic = Critic(dim_state = 512, dim = 256, mlp_depth = 4, dim_latent = 32)

    latent = latent_pool(latent_id = latent_ids, state = state)

    action_distr = actor(state, latent)
    assert isinstance(action_distr, Categorical)
    value = critic(state, latent) # noqa: F841

    # interact with environment and receive rewards, termination etc

    # derive a fitness score for each gene / latent

    fitness = torch.randn(128)

    latent_pool.genetic_algorithm_step(fitness, migrate = num_islands > 1) # update once

    latent_pool.firefly_step(fitness)

@pytest.mark.parametrize(
    'action_is_continuous, distribution_type',
    ((False, Categorical), (True, Beta))
)
def test_actor_returns_distribution(action_is_continuous, distribution_type):
    actor = Actor(
        dim_state = 8,
        dim = 16,
        mlp_depth = 2,
        num_actions = 3,
        action_is_continuous = action_is_continuous
    )

    distr = actor(torch.randn(4, 8), None)

    assert isinstance(distr, distribution_type)
    assert isinstance(distr.sample(), torch.Tensor)

@pytest.mark.parametrize('latent_ids', (2, (2, 4)))
@pytest.mark.parametrize('use_spo', (False, True))
def test_create_agent(
    latent_ids,
    use_spo
):
    from evolutionary_policy_optimization import create_agent

    agent = create_agent(
        dim_state = 512,
        num_latents = 128,
        dim_latent = 32,
        actor_num_actions = 5,
        actor_dim = 256,
        actor_mlp_depth = 2,
        critic_dim = 256,
        critic_mlp_depth = 4,
        wrap_with_accelerate = False,
        use_spo = use_spo
    )

    state = torch.randn(2, 512)

    actions = agent.get_actor_actions(state, latent_id = latent_ids) # noqa: F841
    value = agent.get_critic_values(state, latent_id = latent_ids) # noqa: F841

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
@pytest.mark.parametrize('shrink_and_perturb_every', (None, 1))
def test_e2e_with_mock_env(
    frozen_latents,
    use_critic_ema,
    num_latents,
    diversity_aux_loss_weight,
    critic_use_regression,
    use_improved_critic_loss,
    shrink_and_perturb_every
):
    from evolutionary_policy_optimization import create_agent, EPO, Env

    agent = create_agent(
        dim_state = 512,
        num_latents = num_latents,
        dim_latent = 32,
        actor_num_actions = 5,
        actor_dim = 256,
        actor_mlp_depth = 2,
        critic_dim = 256,
        critic_mlp_depth = 4,
        use_critic_ema = use_critic_ema,
        diversity_aux_loss_weight = diversity_aux_loss_weight,
        shrink_and_perturb_every = shrink_and_perturb_every,
        critic_kwargs = dict(
            use_regression = critic_use_regression
        ),
        use_improved_critic_loss = use_improved_critic_loss,
        latent_gene_pool_kwargs = dict(
            frozen_latents = frozen_latents,
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    epo = EPO(
        agent,
        episodes_per_latent = 1,
        max_episode_length = 10,
        action_sample_temperature = 1.
    )

    env = Env((512,))

    epo(env, num_learning_cycles = 2)

    # saving and loading

    agent.save('./agent.pt', overwrite = True)
    agent.load('./agent.pt')

    shrink_and_perturb_(agent)


def test_beta_entropy_adjusted_for_shift():
    import math
    from torch.distributions import Beta as TorchBeta
    from evolutionary_policy_optimization import BetaActionDistr

    distr_mod = BetaActionDistr()
    params = torch.randn(4, 6, 2)

    entropy = distr_mod.entropy(params)

    # compute raw PyTorch beta entropy for comparison
    mean = distr_mod.mean(params)
    _, raw_conc = params.unbind(dim = -1)
    conc = torch.nn.functional.softplus(raw_conc + distr_mod.raw_init_conc) + distr_mod.min_conc
    conc = conc + 1. / torch.minimum(mean, 1. - mean).clamp(min = distr_mod.eps)
    alpha = mean * conc
    beta = (1. - mean) * conc
    raw_entropy = TorchBeta(alpha, beta).entropy()

    # the adjusted entropy from BetaActionDistr.entropy must equal raw entropy + log(2)
    assert torch.allclose(entropy, raw_entropy + math.log(2.), atol = 1e-5)
    # raw distribution forward returns standard PyTorch Beta
    assert torch.allclose(distr_mod(params).entropy(), raw_entropy, atol = 1e-5)

def test_e2e_with_spr():
    agent = create_agent(
        dim_state = 32,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        use_critic_ema = False,
        use_hidden_spr = True,
        hidden_spr_weight = 1.0,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    epo = EPO(
        agent,
        episodes_per_latent = 2,
        max_episode_length = 8,
    )

    env = Env((32,))
    epo(env, num_learning_cycles = 2)

    agent.save('./agent_spr.pt', overwrite = True)
    agent.load('./agent_spr.pt')


def test_hidden_spr_properties():
    agent = create_agent(
        dim_state = 32,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        use_critic_ema = False,
        use_hidden_spr = True,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    # 1. target networks must have requires_grad = False
    assert all(not p.requires_grad for p in agent.hidden_spr.target_actor.parameters())
    assert all(not p.requires_grad for p in agent.hidden_spr.target_proj.parameters())

    # 2. actor must not be registered as a submodule of hidden_spr (no param aliasing)
    assert 'actor' not in agent.hidden_spr._modules
    params = list(agent.parameters())
    assert len(params) == len(set(params))


def test_e2e_with_spr_discrete():
    agent = create_agent(
        dim_state = 32,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = False,
        use_critic_ema = False,
        use_hidden_spr = True,
        hidden_spr_weight = 1.0,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    epo = EPO(
        agent,
        episodes_per_latent = 2,
        max_episode_length = 8,
    )

    env = Env((32,))
    epo(env, num_learning_cycles = 2)

    agent.save('./agent_spr_discrete.pt', overwrite = True)
    agent.load('./agent_spr_discrete.pt')


def test_value_clipping_defaults():
    agent = create_agent(
        dim_state = 32,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    # value clipping must be off by default
    assert agent.clip_value is False
    # default eps_clip should have lowered aggressiveness (0.8 instead of 0.4)
    assert agent.critic_loss_kwargs.get('eps_clip') == 0.8


def test_e2e_with_value_clipping():
    agent = create_agent(
        dim_state = 32,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        use_critic_ema = False,
        clip_value = True,
        critic_loss_kwargs = dict(
            eps_clip = 0.8,
        ),
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    epo = EPO(
        agent,
        episodes_per_latent = 2,
        max_episode_length = 8,
    )

    env = Env((32,))
    epo(env, num_learning_cycles = 2)


def test_diversity_discr_rollout_no_grad():
    agent = create_agent(
        dim_state = 32,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        use_critic_ema = False,
        use_diversity_discr = True,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    # warm up discriminator
    agent.has_diversity_discr_warmed_up.copy_(torch.tensor(True))

    epo = EPO(
        agent,
        episodes_per_latent = 2,
        max_episode_length = 8,
    )

    env = Env((32,))
    result = epo.gather_experience_from(env)

    # cumulative rewards must not have grad tracking attached
    assert result.cumulative_rewards.grad_fn is None
    assert not result.cumulative_rewards.requires_grad
    for m in result.memories:
        assert m.reward.grad_fn is None
        assert not m.reward.requires_grad


@pytest.mark.parametrize('action_is_continuous', (False, True))
def test_e2e_vectorized_env(action_is_continuous):
    num_latents, episodes_per_latent = 4, 3

    agent = create_agent(
        dim_state = 32,
        num_latents = num_latents,
        dim_latent = 8,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = action_is_continuous,
        wrap_with_accelerate = False,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9,
        ),
    )

    epo = EPO(
        agent,
        episodes_per_latent = episodes_per_latent,
        max_episode_length = 12,
    )

    env = VectorEnv(state_shape = 32, num_envs = 3, can_terminate_after = 2)

    # 1. rollout across parallel workers: check episode counts, contiguity, and shapes
    result = epo.gather_experience_from(env)
    assert result.cumulative_rewards.shape == (num_latents, episodes_per_latent)

    real_memories = [m for m in result.memories if int(m.episode_id) != -1]
    unique_episodes = {(int(m.latent_gene_id), int(m.episode_id)) for m in real_memories}
    assert len(unique_episodes) == num_latents * episodes_per_latent

    # verify contiguity (no interleaving across workers)
    seen, prev = set(), None
    for m in real_memories:
        key = (int(m.latent_gene_id), int(m.episode_id))
        if key != prev:
            assert key not in seen
            seen.add(key)
            prev = key

    # 2. full learning cycle
    epo(env, num_learning_cycles = 2)
    env.close()

    # 3. save and load
    agent.save('./agent_vec.pt', overwrite = True)
    agent.load('./agent_vec.pt')
