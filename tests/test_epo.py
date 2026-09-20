import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, TransformedDistribution

from evolutionary_policy_optimization.epo import EPO, Actor, Critic, LatentGenePool, create_agent, shrink_and_perturb_
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
    ((False, Categorical), (True, TransformedDistribution))
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
    from evolutionary_policy_optimization import EPO, Env, create_agent

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


def test_mean_conc_beta_integration():
    import math

    from evolutionary_policy_optimization import Actor, Beta

    beta = Beta()
    params = torch.randn(4, 6, 2)
    distr = beta(params)

    # entropy directly on distr matches base_dist entropy + log(2)
    base_entropy = distr.base_dist.entropy()
    assert torch.allclose(distr.entropy(), base_entropy + math.log(2.), atol = 1e-5)

    # actor with continuous actions
    actor = Actor(dim_state = 8, dim = 16, mlp_depth = 2, num_actions = 3, action_is_continuous = True)
    distr = actor(torch.randn(4, 8), None)
    sample = distr.sample()
    assert ((sample >= -1.) & (sample <= 1.)).all()
    assert distr.log_prob(sample).shape == (4, 3)
    assert distr.mean.shape == (4, 3)
    assert distr.entropy().shape == (4, 3)

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

def test_diayn_and_dads_diversity_discr():
    from evolutionary_policy_optimization import DiversityDiscr

    # DIAYN: state only (time_gap = 0)
    discr_diayn = DiversityDiscr(
        dim_state = 16,
        num_latents = 4,
        dim = 32,
        depth = 2,
        time_gap = 0
    )

    state = torch.randn(8, 16)
    logits_diayn = discr_diayn(state)
    assert logits_diayn.shape == (8, 4)

    loss_diayn = F.cross_entropy(logits_diayn, torch.randint(0, 4, (8,)))
    loss_diayn.backward()
    assert discr_diayn.state_proj.weight.grad is not None

    # DADS: state transition (time_gap = 1)
    discr_dads = DiversityDiscr(
        dim_state = 16,
        num_latents = 4,
        dim = 32,
        depth = 2,
        time_gap = 1
    )

    next_state = state + torch.randn(8, 16) * 0.1
    logits_dads = discr_dads(state, next_state)
    assert logits_dads.shape == (8, 4)

    loss_dads = F.cross_entropy(logits_dads, torch.randint(0, 4, (8,)))
    loss_dads.backward()
    assert discr_dads.state_proj.weight.grad is not None


@pytest.mark.parametrize('diversity_time_gap', (0, 1, 2))
def test_e2e_learning_cycle_with_diversity_discr(diversity_time_gap):
    agent = create_agent(
        dim_state = 16,
        num_latents = 4,
        dim_latent = 8,
        actor_num_actions = 2,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        use_critic_ema = False,
        use_diversity_discr = True,
        diversity_time_gap = diversity_time_gap,
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.75,
            frac_tournaments = 0.9
        ),
        wrap_with_accelerate = False,
    )

    epo = EPO(
        agent,
        episodes_per_latent = 2,
        max_episode_length = 6,
        diversity_reward_weight = 0.5,
    )

    env = Env((16,))
    # after 1 cycle, discriminator should be warmed up
    epo(env, num_learning_cycles = 1)
    assert agent.has_diversity_discr_warmed_up.item()

    # after 2nd cycle, GA step runs (apply_genetic_algorithm_every=2) which resets warmup
    epo(env, num_learning_cycles = 1)
    assert not agent.has_diversity_discr_warmed_up.item()


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


def test_continuous_beta_details():
    from evolutionary_policy_optimization import Beta, create_agent

    agent = create_agent(
        dim_state = 16,
        num_latents = 64,
        dim_latent = 8,
        actor_num_actions = 3,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        wrap_with_accelerate = False,
    )

    state = torch.randn(4, 16)

    # 1. greedy action (sample = False)
    actions = agent.get_actor_actions(state, latent_id = 0, sample = False)
    assert actions.shape == (4, 3)
    assert ((actions >= -1.) & (actions <= 1.)).all()

    # 2. sampled action (sample = True)
    actions_sample, log_probs = agent.get_actor_actions(state, latent_id = 0, sample = True)
    assert actions_sample.shape == (4, 3)
    assert log_probs.shape == (4,)
    assert ((actions_sample >= -1.) & (actions_sample <= 1.)).all()

    # 3. temperature support
    actions_temp, _ = agent.get_actor_actions(state, latent_id = 0, sample = True, temperature = 0.5)
    assert actions_temp.shape == (4, 3)

    # 4. scaling to env bounds by multiplying constant
    scaled = actions * 0.4
    assert scaled.shape == (4, 3)
    assert ((scaled >= -0.4) & (scaled <= 0.4)).all()

    # 5. beta_kwargs pass-through
    agent_custom = create_agent(
        dim_state = 16,
        num_latents = 64,
        dim_latent = 8,
        actor_num_actions = 3,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        actor_kwargs = dict(beta_kwargs = dict(pos_fn = 'softplus')),
        wrap_with_accelerate = False,
    )
    assert agent_custom.actor.action_distr.pos_fn == 'softplus'
