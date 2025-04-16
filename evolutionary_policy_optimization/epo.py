from __future__ import annotations

from collections import namedtuple

import torch
from torch import nn, cat, is_tensor, tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader

import einx
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange

from assoc_scan import AssocScan

from adam_atan2_pytorch import AdoptAtan2

from hl_gauss_pytorch import HLGaussLayer

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def calc_entropy(logits):
    prob = logits.softmax(dim = -1)
    return -prob * log(prob)

def gather_log_prob(
    logits, # Float[b l]
    indices # Int[b]
): # Float[b]
    indices = rearrange(indices, '... -> ... 1')
    log_probs = logits.log_softmax(dim = -1)
    log_prob = log_probs.gather(-1, indices)
    return rearrange(log_prob, '... 1 -> ...')

# generalized advantage estimate

def calc_generalized_advantage_estimate(
    rewards, # Float[g n]
    values,  # Float[g n+1]
    masks,   # Bool[n]
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None

):
    assert values.shape[-1] == (rewards.shape[-1] + 1)

    use_accelerated = default(use_accelerated, rewards.is_cuda)
    device = rewards.device

    masks = repeat(masks, 'n -> g n', g = rewards.shape[0])

    values, values_next = values[:, :-1], values[:, 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    gates, delta = gates[..., :, None], delta[..., :, None]

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    gae = gae[..., :, 0]

    return gae

# evolution related functions

def crossover_latents(
    parent1, parent2,
    weight = None,
    random = False,
    l2norm_output = False
):
    assert parent1.shape == parent2.shape

    if random:
        assert not exists(weight)
        weight = torch.randn_like(parent1).sigmoid()
    else:
        weight = default(weight, 0.5) # they do a simple averaging for the latents as crossover, but allow for random interpolation, as well extend this work for tournament selection, where same set of parents may be re-selected

    child = torch.lerp(parent1, parent2, weight)

    if not l2norm_output:
        return child

    return l2norm(child)

def mutation(
    latents,
    mutation_strength = 1.,
    l2norm_output = False
):
    mutations = torch.randn_like(latents)

    mutated = latents + mutations * mutation_strength

    if not l2norm_output:
        return mutated

    return l2norm(mutated)

# simple MLP networks, but with latent variables
# the latent variables are the "genes" with the rest of the network as the scaffold for "gene expression" - as suggested in the paper

class MLP(Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        dim_latent = 0,
    ):
        super().__init__()
        assert len(dims) >= 2, 'must have at least two dimensions'

        # add the latent to the first dim

        first_dim, *rest_dims = dims
        dims = (first_dim + dim_latent, *rest_dims)

        self.dim_latent = dim_latent

        self.needs_latent = dim_latent > 0

        self.encode_latent = nn.Sequential(
            Linear(dim_latent, dim_latent),
            nn.SiLU()
        ) if self.needs_latent else None

        # pairs of dimension

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # modules across layers

        layers = ModuleList([Linear(dim_in, dim_out) for dim_in, dim_out in dim_pairs])

        self.layers = layers

    def forward(
        self,
        x,
        latent = None
    ):
        batch = x.shape[0]

        assert xnor(self.needs_latent, exists(latent))

        if exists(latent):
            # start with naive concatenative conditioning
            # but will also offer some alternatives once a spark is seen (film, adaptive linear from stylegan, etc)

            latent = self.encode_latent(latent)

            if latent.ndim == 1:
                latent = repeat(latent, 'd -> b d', b = batch)

            assert latent.shape[0] == x.shape[0], f'received state with batch size {x.shape[0]} but latent ids received had batch size {latent_id.shape[0]}'

            x = cat((x, latent), dim = -1)

        # layers

        for ind, layer in enumerate(self.layers, start = 1):
            is_last = ind == len(self.layers)

            x = layer(x)

            if not is_last:
                x = F.silu(x)

        return x

# actor, critic, and agent (actor + critic)
# eventually, should just create a separate repo and aggregate all the MLP related architectures

class Actor(Module):
    def __init__(
        self,
        dim_state,
        num_actions,
        dim_hiddens: tuple[int, ...],
        dim_latent = 0,
    ):
        super().__init__()

        assert len(dim_hiddens) >= 2
        dim_first, *_, dim_last = dim_hiddens

        self.dim_latent = dim_latent

        self.init_layer = nn.Sequential(
            nn.Linear(dim_state, dim_first),
            nn.SiLU()
        )

        self.mlp = MLP(dims = dim_hiddens, dim_latent = dim_latent)

        self.to_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_last, num_actions),
        )

    def forward(
        self,
        state,
        latent
    ):

        hidden = self.init_layer(state)

        hidden = self.mlp(hidden, latent)

        return self.to_out(hidden)

class Critic(Module):
    def __init__(
        self,
        dim_state,
        dim_hiddens: tuple[int, ...],
        dim_latent = 0,
    ):
        super().__init__()

        assert len(dim_hiddens) >= 2
        dim_first, *_, dim_last = dim_hiddens

        self.dim_latent = dim_latent

        self.init_layer = nn.Sequential(
            nn.Linear(dim_state, dim_first),
            nn.SiLU()
        )

        self.mlp = MLP(dims = dim_hiddens, dim_latent = dim_latent)

        self.to_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_last, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        state,
        latent
    ):

        hidden = self.init_layer(state)

        hidden = self.mlp(hidden, latent)

        return self.to_out(hidden)

# criteria for running genetic algorithm

class ShouldRunGeneticAlgorithm(Module):
    def __init__(
        self,
        gamma = 1.5 # not sure what the value is
    ):
        super().__init__()
        self.gamma = gamma

    def forward(self, fitnesses):
        # equation (3)

        # max(fitness) - min(fitness) > gamma * median(fitness)
        # however, this equation does not make much sense to me if fitness increases unbounded
        # just let it be customizable, and offer a variant where mean and variance is over some threshold (could account for skew too)

        return (fitnesses.amax(dim = -1) - fitnesses.amin(dim = -1)) > (self.gamma * torch.median(fitnesses, dim = -1).values)

# classes

class LatentGenePool(Module):
    def __init__(
        self,
        num_latents,                     # same as gene pool size
        dim_latent,                      # gene dimension
        num_latent_sets = 1,             # allow for sets of latents / gene per individual, expression of a set controlled by the environment
        num_islands = 1,                 # add the island strategy, which has been effectively used in a few recent works
        dim_state = None,
        frozen_latents = True,
        crossover_random = True,         # random interp from parent1 to parent2 for crossover, set to `False` for averaging (0.5 constant value)
        l2norm_latent = False,           # whether to enforce latents on hypersphere,
        frac_tournaments = 0.25,         # fraction of genes to participate in tournament - the lower the value, the more chance a less fit gene could be selected
        frac_natural_selected = 0.25,    # number of least fit genes to remove from the pool
        frac_elitism = 0.1,              # frac of population to preserve from being noised
        mutation_strength = 1.,          # factor to multiply to gaussian noise as mutation to latents
        should_run_genetic_algorithm: Module | None = None, # eq (3) in paper
        default_should_run_ga_gamma = 1.5
    ):
        super().__init__()

        maybe_l2norm = l2norm if l2norm_latent else identity

        latents = torch.randn(num_latents, num_latent_sets, dim_latent)

        if l2norm_latent:
            latents = maybe_l2norm(latents, dim = -1)

        self.num_latents = num_latents
        self.needs_latent_gate = num_latent_sets > 1
        self.latents = nn.Parameter(latents, requires_grad = not frozen_latents)

        self.maybe_l2norm = maybe_l2norm

        # gene expression as a function of environment

        self.num_latent_sets = num_latent_sets

        if self.needs_latent_gate:
            assert exists(dim_state), '`dim_state` must be passed in if using gated gene expression'

        self.to_latent_gate = nn.Sequential(
            Linear(dim_state, num_latent_sets),
            nn.Softmax(dim = -1)
        ) if self.needs_latent_gate else None

        # some derived values

        assert num_islands >= 1
        assert divisible_by(num_latents, num_islands)

        assert 0. < frac_tournaments < 1.
        assert 0. < frac_natural_selected < 1.
        assert 0. <= frac_elitism < 1.
        assert (frac_natural_selected + frac_elitism) < 1.

        self.dim_latent = dim_latent
        self.num_latents = num_latents
        self.num_islands = num_islands

        latents_per_island = num_latents // num_islands
        self.num_natural_selected = int(frac_natural_selected * latents_per_island)

        self.num_tournament_participants = int(frac_tournaments * self.num_natural_selected)
        self.crossover_random  = crossover_random

        self.mutation_strength = mutation_strength
        self.num_elites = int(frac_elitism * latents_per_island)
        self.has_elites = self.num_elites > 0

        if not exists(should_run_genetic_algorithm):
            should_run_genetic_algorithm = ShouldRunGeneticAlgorithm(gamma = default_should_run_ga_gamma)

        self.should_run_genetic_algorithm = should_run_genetic_algorithm

    def firefly_step(
        self,
        fitness,
        beta0 = 2.,           # exploitation factor, moving fireflies of low light intensity to high
        gamma = 1.,           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
        alpha = 0.1,          # exploration factor
        alpha_decay = 0.995,  # exploration decay each step
        inplace = True,
    ):
        islands = self.num_islands
        fireflies = self.latents # the latents are the fireflies

        assert fitness.shape[0] == fireflies.shape[0]

        fitness = rearrange(fitness, '(i p) -> i p', i = islands)
        fireflies = rearrange(fireflies, '(i p) ... -> i p ...', i = islands)

        # fireflies with lower light intensity (high cost) moves towards the higher intensity (lower cost)

        move_mask = einx.less('i x, i y -> i x y', fitness, fitness)

        # get vectors of fireflies to one another
        # calculate distance and the beta

        delta_positions = einx.subtract('i y ... d, i x ... d -> i x y ... d', fireflies, fireflies)

        distance = delta_positions.norm(dim = -1)

        betas = beta0 * (-gamma * distance ** 2).exp()

        # move the fireflies according to attraction

        fireflies += einsum(move_mask, betas, delta_positions, 'i x y, i x y ..., i x y ... -> i x ...')

        # merge back the islands

        fireflies = rearrange(fireflies, 'i p ... -> (i p) ...')

        if not inplace:
            return fireflies

        self.latents.copy_(fireflies)

    @torch.no_grad()
    # non-gradient optimization, at least, not on the individual level (taken care of by rl component)
    def genetic_algorithm_step(
        self,
        fitness, # Float['p'],
        inplace = True
    ):
        """
        i - islands
        p - population
        g - gene dimension
        n - number of genes per individual
        t - num tournament participants
        """

        islands = self.num_islands
        tournament_participants = self.num_tournament_participants

        assert self.num_latents > 1

        genes = self.latents # the latents are the genes

        pop_size = genes.shape[0]
        assert pop_size == fitness.shape[0]

        pop_size_per_island = pop_size // islands

        # split out the islands

        fitness = rearrange(fitness, '(i p) -> i p', i = islands)

        # from the fitness, decide whether to actually run the genetic algorithm or not

        should_update_per_island = self.should_run_genetic_algorithm(fitness)

        if not should_update_per_island.any():
            if inplace:
                return

            return genes

        genes = rearrange(genes, '(i p) n g -> i p n g', i = islands)

        orig_genes = genes

        # 1. natural selection is simple in silico
        # you sort the population by the fitness and slice off the least fit end

        sorted_indices = fitness.sort(dim = -1).indices
        natural_selected_indices = sorted_indices[..., -self.num_natural_selected:]
        natural_select_gene_indices = repeat(natural_selected_indices, '... -> ... n g', n = genes.shape[-2], g = genes.shape[-1])

        genes, fitness = genes.gather(1, natural_select_gene_indices), fitness.gather(1, natural_selected_indices)

        # 2. for finding pairs of parents to replete gene pool, we will go with the popular tournament strategy

        rand_tournament_gene_ids = torch.randn((islands, pop_size_per_island - self.num_natural_selected, tournament_participants)).argsort(dim = -1)
        rand_tournament_gene_ids_for_gather = rearrange(rand_tournament_gene_ids, 'i p t -> i (p t)')

        participant_fitness = fitness.gather(1, rand_tournament_gene_ids_for_gather)
        participant_fitness = rearrange(participant_fitness, 'i (p t) -> i p t', t = tournament_participants)

        parent_indices_at_tournament = participant_fitness.topk(2, dim = -1).indices
        parent_gene_ids = rand_tournament_gene_ids.gather(-1, parent_indices_at_tournament)

        parent_gene_ids_for_gather = repeat(parent_gene_ids, 'i p parents -> i (p parents) n g', n = genes.shape[-2], g = genes.shape[-1])

        parents = genes.gather(1, parent_gene_ids_for_gather)
        parents = rearrange(parents, 'i (p parents) ... -> i p parents ...', parents = 2)

        # 3. do a crossover of the parents - in their case they went for a simple averaging, but since we are doing tournament style and the same pair of parents may be re-selected, lets make it random interpolation

        parent1, parent2 = parents.unbind(dim = 2)
        children = crossover_latents(parent1, parent2, random = self.crossover_random)

        # append children to gene pool

        genes = cat((children, genes), dim = 1)

        # 4. they use the elitism strategy to protect best performing genes from being changed

        if self.has_elites:
            genes, elites = genes[:, :-self.num_elites], genes[:, -self.num_elites:]

        # 5. mutate with gaussian noise - todo: add drawing the mutation rate from exponential distribution, from the fast genetic algorithms paper from 2017

        genes = mutation(genes, mutation_strength = self.mutation_strength)

        # add back the elites

        if self.has_elites:
            genes = cat((genes, elites), dim = 1)

        genes = self.maybe_l2norm(genes)

        # account for criteria of whether to actually run GA or not

        genes = einx.where('i, i ..., i ...', should_update_per_island, genes, orig_genes)

        # merge island back into pop dimension

        genes = rearrange(genes, 'i p ... -> (i p) ...')

        if not inplace:
            return genes

        # store the genes for the next interaction with environment for new fitness values (a function of reward and other to be researched measures)

        self.latents.copy_(genes)

    def forward(
        self,
        *args,
        state: Tensor | None = None,
        latent_id: int | None = None,
        net: Module | None = None,
        **kwargs,
    ):
        device = self.latents.device

        # if only 1 latent, assume doing ablation and get lone gene

        if not exists(latent_id) and self.num_latents == 1:
            latent_id = 0

        if not is_tensor(latent_id):
            latent_id = tensor(latent_id, device = device)

        assert (0 <= latent_id).all() and (latent_id < self.num_latents).all()

        # fetch latent

        fetching_multiple_latents = latent_id.numel() > 1

        latent = self.latents[latent_id]

        if self.needs_latent_gate:
            assert exists(state), 'state must be passed in if greater than number of 1 latent set'

            if not fetching_multiple_latents:
                latent = repeat(latent, '... -> b ...', b = state.shape[0])

            assert latent.shape[0] == state.shape[0]

            gates = self.to_latent_gate(state)
            latent = einsum(latent, gates, 'b n g, b n -> b g')

        elif fetching_multiple_latents:
            latent = latent[:, 0]
        else:
            latent = latent[0]

        latent = self.maybe_l2norm(latent)

        if not exists(net):
            return latent

        return net(
            *args,
            latent = latent,
            **kwargs
        )

# agent class

class Agent(Module):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        latent_gene_pool: LatentGenePool,
        optim_klass = AdoptAtan2,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        latent_lr = 1e-5,
        actor_optim_kwargs: dict = dict(),
        critic_optim_kwargs: dict = dict(),
        latent_optim_kwargs: dict = dict(),
    ):
        super().__init__()

        self.actor = actor
        self.critic = critic

        self.latent_gene_pool = latent_gene_pool

        assert actor.dim_latent == critic.dim_latent == latent_gene_pool.dim_latent

        # optimizers

        self.actor_optim = optim_klass(actor.parameters(), lr = actor_lr, **actor_optim_kwargs)
        self.critic_optim = optim_klass(critic.parameters(), lr = critic_lr, **critic_optim_kwargs)

        self.latent_optim = optim_klass(latent_gene_pool.parameters(), lr = latent_lr, **latent_optim_kwargs) if latent_gene_pool.needs_latent_gate else None

    def get_actor_actions(
        self,
        state,
        latent_id
    ):
        latent = self.latent_gene_pool(latent_id = latent_id, state = state)
        return self.actor(state, latent)

    def get_critic_values(
        self,
        state,
        latent_id
    ):
        latent = self.latent_gene_pool(latent_id = latent_id, state = state)
        return self.critic(state, latent)

    def update_latent_gene_pool_(
        self,
        fitnesses
    ):
        return self.latent_gene_pool.genetic_algorithm_step(fitnesses)

    def forward(
        self,
        memories: list[Memory]
    ):
        raise NotImplementedError

# reinforcement learning related - ppo

def actor_loss(
    logits,         # Float[b l]
    old_log_probs,  # Float[b]
    actions,        # Int[b]
    advantages,     # Float[b]
    eps_clip = 0.2,
    entropy_weight = .01,
):
    log_probs = gather_log_prob(logits, actions)

    ratio = (log_probs - old_log_probs).exp()

    # classic clipped surrogate loss from ppo

    clipped_ratio = ratio.clamp(min = 1. - eps_clip, max = 1. + eps_clip)

    actor_loss = -torch.min(clipped_ratio * advantage, ratio * advantage)

    # add entropy loss for exploration

    entropy = calc_entropy(logits)

    entropy_aux_loss = -entropy_weight * entropy

    return actor_loss + entropy_aux_loss

def critic_loss(
    pred_values,  # Float[b]
    advantages,   # Float[b]
    old_values    # Float[b]
):
    discounted_values = advantages + old_values
    return F.mse_loss(pred_values, discounted_values)

# agent contains the actor, critic, and the latent genetic pool

def create_agent(
    dim_state,
    num_latents,
    dim_latent,
    actor_num_actions,
    actor_dim_hiddens: int | tuple[int, ...],
    critic_dim_hiddens: int | tuple[int, ...],
    num_latent_sets = 1
) -> Agent:

    actor = Actor(
        num_actions = actor_num_actions,
        dim_state = dim_state,
        dim_latent = dim_latent,
        dim_hiddens = actor_dim_hiddens
    )

    critic = Critic(
        dim_state = dim_state,
        dim_latent = dim_latent,
        dim_hiddens = critic_dim_hiddens
    )

    latent_gene_pool = LatentGenePool(
        dim_state = dim_state,
        num_latents = num_latents,
        dim_latent = dim_latent,
        num_latent_sets = num_latent_sets
    )

    return Agent(actor = actor, critic = critic, latent_gene_pool = latent_gene_pool)

# EPO - which is just PPO with natural selection of a population of latent variables conditioning the agent
# the tricky part is that the latent ids for each episode / trajectory needs to be tracked

Memory = namedtuple('Memory', [
    'state',
    'latent_gene_id',
    'action',
    'log_prob',
    'reward',
    'values',
    'done'
])

class EPO(Module):

    def __init__(
        self,
        agent: Agent
    ):
        super().__init__()
        self.agent = agent

    def forward(
        self,
        env
    ) -> list[Memory]:

        raise NotImplementedError
