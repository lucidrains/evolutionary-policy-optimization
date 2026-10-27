import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, TransformedDistribution

from evolutionary_policy_optimization.epo import EPO, Actor, Critic, LatentGenePool, ShouldRunGeneticAlgorithm, ShuffledBatches, create_agent, shrink_and_perturb_
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
        dim_state = 32,
        num_latents = num_latents,
        dim_latent = 32,
        actor_num_actions = 5,
        actor_dim = 32,
        actor_mlp_depth = 2,
        critic_dim = 32,
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

    env = Env((32,))

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
            frac_tournaments = 0.9,
            should_run_genetic_algorithm = ShouldRunGeneticAlgorithm(gamma = 0., min_spread = -1.)
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

    # next_state of each step is the state the episode carries on from
    for prev, curr in zip(real_memories, real_memories[1:]):
        if (int(prev.latent_gene_id), int(prev.episode_id)) == (int(curr.latent_gene_id), int(curr.episode_id)):
            assert torch.equal(prev.next_state, curr.state)

    # 2. full learning cycle
    epo(env, num_learning_cycles = 2)
    env.close()

    # 3. save and load
    agent.save('./agent_vec.pt', overwrite = True)
    agent.load('./agent_vec.pt')


@pytest.mark.parametrize('action_is_continuous', (False, True))
def test_diversity_discr_continuous_dads(action_is_continuous):
    from evolutionary_policy_optimization import DiversityDiscr

    discr = DiversityDiscr(
        dim_state = 16,
        dim_latent = 6,
        dim = 32,
        depth = 2,
        time_gap = 1,
        dim_action = 4,
        action_is_continuous = action_is_continuous,
    )

    state = torch.randn(8, 16)
    next_state = state + torch.randn(8, 16) * 0.1
    z = torch.randn(8, 6)
    action = torch.randn(8, 4) if action_is_continuous else torch.randint(0, 4, (8,))

    pred = discr(state, next_state, action = action)
    assert pred.shape == (8, 6)

    loss = F.mse_loss(pred, z)
    loss.backward()
    assert discr.state_proj.weight.grad is not None
    assert discr.action_proj.weight.grad is not None

def test_fixed_latent_breaks_symmetry():
    from evolutionary_policy_optimization import create_agent

    agent = create_agent(
        dim_state = 32,
        num_latents = 8,
        dim_latent = 16,
        actor_num_actions = 6,
        actor_dim = 32,
        actor_mlp_depth = 2,
        critic_dim = 32,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        latent_mode = 'fixed',
        use_critic_ema = False,
        wrap_with_accelerate = False,
    )

    state = torch.randn(64, 32)
    z1, z2 = torch.zeros(64, 16), torch.ones(64, 16)

    mean1 = agent.get_actor_distribution(state, latent = z1).mean
    mean2 = agent.get_actor_distribution(state, latent = z2).mean

    assert (mean1 - mean2).abs().mean() > 1e-3

@pytest.mark.parametrize('latent_mode', ('gene_pool', 'fixed'))
def test_epo_latent_mode_toggle(latent_mode):
    # original EPO (evolved gene pool) vs fixed latents sampled from N(0, I) + DADS

    from evolutionary_policy_optimization import EPO, create_agent

    fixed = latent_mode == 'fixed'
    num_latents = 8 if fixed else 4

    agent = create_agent(
        dim_state = 16,
        num_latents = num_latents,
        dim_latent = 6,
        actor_num_actions = 4,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        latent_mode = latent_mode,
        use_diversity_discr = fixed,
        use_critic_ema = False,
        wrap_with_accelerate = False,
        latent_gene_pool_kwargs = dict(frac_natural_selected = 0.75, frac_tournaments = 0.9),
    )

    assert agent.has_latent_genes is (not fixed)
    assert agent.has_fixed_latents is fixed
    assert agent.use_diversity_discr is fixed

    # fixed mode has no gene ids - z must come from the sampler

    if fixed:
        with pytest.raises(AssertionError):
            agent.get_actor_actions(torch.randn(2, 16), latent_id = 0)

    epo = EPO(agent, episodes_per_latent = 1, max_episode_length = 6, diversity_reward_weight = 0.5)
    env = VectorEnv(state_shape = 16, num_envs = 8, can_terminate_after = 2)

    memories = epo.gather_experience_from(env)
    assert memories.cumulative_rewards.shape == (num_latents, 1)

    real = [m for m in memories.memories if int(m.episode_id) != -1]
    refs = {(tuple(m.latent_gene_id.tolist()) if fixed else int(m.latent_gene_id)) for m in real}
    assert len(refs) == num_latents
    assert all(m.latent_gene_id.shape == ((6,) if fixed else ()) for m in real)

    # one cycle warms the discriminator, the second trains against DADS rewards

    epo(env, num_learning_cycles = 2)
    assert agent.has_diversity_discr_warmed_up.item() is fixed

    agent.save('./agent_latent_mode.pt', overwrite = True)
    agent.load('./agent_latent_mode.pt')

def test_continuous_beta_details():
    from evolutionary_policy_optimization import create_agent

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
    assert log_probs.shape == (4, 3)
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


@pytest.mark.parametrize('num_transitions, batch_size', ((32, 8), (30, 8), (1, 8), (8, 8)))
def test_shuffled_batches(num_transitions, batch_size):
    ids = torch.arange(num_transitions)
    batches = ShuffledBatches((ids, -ids), batch_size)

    for _ in range(2): # a fresh full pass on every iteration
        gathered, batch_sizes = [], []

        for batch_ids, batch_neg_ids in batches:
            assert torch.equal(batch_ids, -batch_neg_ids) # fields stay row aligned
            assert batch_ids.numel() <= batch_size

            gathered.append(batch_ids)
            batch_sizes.append(batch_ids.numel())

        gathered = torch.cat(gathered)

        assert gathered.sort().values.tolist() == ids.tolist() # every sample exactly once per pass
        assert batch_sizes[-1] == (num_transitions - 1) % batch_size + 1 # only the last batch can be short


@pytest.mark.parametrize('epochs', (1, 3))
def test_learn_from_shuffles_each_epoch(monkeypatch, epochs):
    import evolutionary_policy_optimization.epo as epo_module
    from evolutionary_policy_optimization.epo import MemoriesAndCumulativeRewards, Memory

    dim_state, num_actions, num_transitions, batch_size = 8, 4, 20, 8

    agent = create_agent(
        dim_state = dim_state,
        num_latents = 1,
        dim_latent = 4,
        actor_num_actions = num_actions,
        actor_dim = 8,
        actor_mlp_depth = 2,
        critic_dim = 8,
        critic_mlp_depth = 2,
        wrap_with_accelerate = False,
        batch_size = batch_size,
    )

    states = torch.randn(num_transitions, dim_state)

    with torch.no_grad():
        actions, log_probs = agent.get_actor_actions(states, sample = True, use_unwrapped_model = True)

    memories = [
        Memory(
            episode_id = torch.tensor(i % 2),
            state = states[i],
            next_state = torch.randn(dim_state),
            latent_gene_id = torch.tensor(0),
            action = actions[i],
            log_prob = log_probs[i],
            reward = torch.tensor(0.),
            value = torch.tensor(0.),
            done = torch.tensor(False)
        )
        for i in range(num_transitions)
    ]

    num_passes = []

    class CountingShuffledBatches(epo_module.ShuffledBatches):
        def __iter__(self):
            num_passes.append(1)
            yield from super().__iter__()

    monkeypatch.setattr(epo_module, 'ShuffledBatches', CountingShuffledBatches)

    agent.learn_from(MemoriesAndCumulativeRewards(memories, torch.zeros(1, 2)), epochs = epochs)

    assert len(num_passes) == epochs # a fresh shuffled pass per epoch


@pytest.mark.parametrize('per_action_dim', (False, True))
def test_hpo_per_action_dim_continuous(per_action_dim):
    agent = create_agent(
        dim_state = 16,
        num_latents = 1,
        dim_latent = 0,
        actor_num_actions = 6,
        actor_dim = 16,
        actor_mlp_depth = 2,
        critic_dim = 16,
        critic_mlp_depth = 2,
        action_is_continuous = True,
        use_critic_ema = False,
        actor_loss_kwargs = dict(
            use_hpo = True,
            per_action_dim = per_action_dim,
            eps_clip = 0.2,
            entropy_weight = 0.002,
            norm_advantages = True,
        ),
        wrap_with_accelerate = False,
    )

    epo = EPO(agent, episodes_per_latent = 4, max_episode_length = 8)
    env = VectorEnv(state_shape = 16, num_envs = 4, can_terminate_after = 3)

    rollout = epo.gather_experience_from(env)
    real = [m for m in rollout.memories if int(m.episode_id) != -1]
    assert all(m.log_prob.shape == (6,) for m in real)

    agent.learn_from(rollout, epochs = 2)
    env.close()
