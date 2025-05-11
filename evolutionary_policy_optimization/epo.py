from __future__ import annotations
from typing import Callable

from pathlib import Path
from math import ceil
from itertools import product
from functools import partial, wraps
from collections import namedtuple
from random import randrange

import numpy as np

import torch
from torch import nn, cat, stack, is_tensor, tensor, from_numpy, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Linear, Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader
from torch.utils._pytree import tree_map

import einx
from einops import rearrange, repeat, reduce, einsum, pack
from einops.layers.torch import Rearrange

from evolutionary_policy_optimization.distributed import (
    is_distributed,
    get_world_and_rank,
    maybe_sync_seed,
    all_gather_variable_dim,
    maybe_barrier
)

from assoc_scan import AssocScan

from adam_atan2_pytorch import AdoptAtan2

from hl_gauss_pytorch import HLGaussLayer

from ema_pytorch import EMA

from tqdm import tqdm

from accelerate import Accelerator

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

def to_device(inp, device):
    return tree_map(lambda t: t.to(device) if is_tensor(t) else t, inp)

def maybe(fn):

    @wraps(fn)
    def decorated(inp, *args, **kwargs):
        if not exists(inp):
            return None

        return fn(inp, *args, **kwargs)

    return decorated

def interface_torch_numpy(fn, device):
    # for a given function, move all inputs from torch tensor to numpy, and all outputs from numpy to torch tensor

    @maybe
    def to_torch_tensor(t):
        if isinstance(t, (np.ndarray, np.float64)):
            t = from_numpy(np.array(t))
        elif isinstance(t, (float, int, bool)):
            t = tensor(t)

        return t.to(device)

    @wraps(fn)
    def decorated_fn(*args, **kwargs):

        args, kwargs = tree_map(lambda t: t.cpu().numpy() if is_tensor(t) else t, (args, kwargs))

        out = fn(*args, **kwargs)

        out = tree_map(to_torch_tensor, out)
        return out

    return decorated_fn

def move_input_tensors_to_device(fn):

    @wraps(fn)
    def decorated_fn(self, *args, **kwargs):
        args, kwargs = tree_map(lambda t: t.to(self.device) if is_tensor(t) else t, (args, kwargs))

        return fn(self, *args, **kwargs)

    return decorated_fn

# tensor helpers

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def batch_randperm(shape, device):
    return torch.randn(shape, device = device).argsort(dim = -1)

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(t, temperature = 1.):
    is_greedy = temperature <= 0.

    if not is_greedy:
        t = (t / temperature) + gumbel_noise(t)

    return t.argmax(dim = -1)

def calc_entropy(logits):
    prob = logits.softmax(dim = -1)
    return -(prob * log(prob)).sum(dim = -1)

def gather_log_prob(
    logits, # Float[b l]
    indices # Int[b]
): # Float[b]
    indices = rearrange(indices, '... -> ... 1')
    log_probs = logits.log_softmax(dim = -1)
    log_prob = log_probs.gather(-1, indices)
    return rearrange(log_prob, '... 1 -> ...')

def temp_batch_dim(fn):

    @wraps(fn)
    def inner(*args, **kwargs):
        args, kwargs = tree_map(lambda t: rearrange(t, '... -> 1 ...') if is_tensor(t) else t, (args, kwargs))

        out = fn(*args, **kwargs)

        out = tree_map(lambda t: rearrange(t, '1 ... -> ...') if is_tensor(t) else t, out)
        return out

    return inner

# plasticity related

def shrink_and_perturb_(
    module,
    shrink_factor = 0.5,
    perturb_factor = 0.01
):
    # Shrink & Perturb
    # Ash et al. https://arxiv.org/abs/1910.08475

    assert 0. <= shrink_factor <= 1.

    for p in module.parameters():
        noise = torch.randn_like(p.data)
        p.data.mul_(1. - shrink_factor).add_(noise * perturb_factor)

    return module

# fitness related

def get_fitness_scores(
    cum_rewards, # Float['gene episodes']
    memories
): # Float['gene']
    return cum_rewards.sum(dim = -1) # sum all rewards across episodes, but could override this function for normalizing with whatever

# generalized advantage estimate

def calc_generalized_advantage_estimate(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[:-1], values[1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    return scan(gates, delta)

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

    if is_tensor(mutation_strength):
        mutations = einx.multiply('b, b ...', mutation_strength, mutations)
    else:
        mutations *= mutation_strength

    mutated = latents + mutations

    if not l2norm_output:
        return mutated

    return l2norm(mutated)

# drawing mutation strengths from power law distribution
# proposed by https://arxiv.org/abs/1703.03334

class PowerLawDist(Module):
    def __init__(
        self,
        values: Tensor | list[float] | None = None,
        bins = None,
        beta = 1.5,
    ):
        super().__init__()
        assert beta > 1.

        assert exists(bins) or exists(values)

        if exists(values):
            if not is_tensor(values):
                values = tensor(values)

            assert values.ndim == 1
            bins = values.shape[0]

        self.beta = beta

        cdf = torch.linspace(1, bins, bins).pow(-beta).cumsum(dim = -1)
        cdf = cdf / cdf[-1]

        self.register_buffer('cdf', cdf)
        self.register_buffer('values', values)

    def forward(self, shape):
        device = self.cdf.device

        uniform = torch.rand(shape, device = device)

        sampled = torch.searchsorted(self.cdf, uniform)

        if not exists(self.values):
            return sampled

        return self.values[sampled]

# FiLM for latent to mlp conditioning

class FiLM(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.to_gamma = nn.Linear(dim, dim_out, bias = False)
        self.to_beta = nn.Linear(dim, dim_out, bias = False)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_beta.weight)

    def forward(self, x, cond):
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)

        return x * (gamma + 1.) + beta

# layer integrated memory

class DynamicLIMe(Module):
    def __init__(
        self,
        dim,
        num_layers
    ):
        super().__init__()
        self.num_layers = num_layers

        self.to_weights = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_layers),
            nn.Softmax(dim = -1)
        )

    def forward(
        self,
        x,
        hiddens
    ):

        if not is_tensor(hiddens):
            hiddens = stack(hiddens)

        assert hiddens.shape[0] == self.num_layers, f'expected hiddens to have {self.num_layers} layers but received {tuple(hiddens.shape)} instead (first dimension must be layers)'

        weights = self.to_weights(x)

        return einsum(hiddens, weights, 'l b d, b l -> b d')

# state normalization

class StateNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754 - 'RSMNorm'

        super().__init__()
        self.dim = dim
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def forward(
        self,
        state
    ):
        assert state.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        time = self.step.item()
        mean = self.running_mean
        variance = self.running_variance

        normed = (state - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        new_obs_mean = reduce(state, '... d -> d', 'mean')
        delta = new_obs_mean - mean

        new_mean = mean + delta / time
        new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

        self.step.add_(1)
        self.running_mean.copy_(new_mean)
        self.running_variance.copy_(new_variance)

        return normed

# style mapping network from StyleGAN2

class EqualLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        lr_mul = 1,
        bias = True
    ):
        super().__init__()
        self.lr_mul = lr_mul

        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))
        self.bias = nn.Parameter(torch.zeros(dim_out))

    def forward(
        self,
        input
    ):
        weight, bias = tuple(t * self.lr_mul for t in (self.weight, self.bias))
        return F.linear(input, weight, bias = bias)

class LatentMappingNetwork(Module):
    def __init__(
        self,
        dim_latent,
        depth,
        lr_mul = 0.1,
        leaky_relu_p = 2e-2
    ):
        super().__init__()

        layers = []

        for i in range(depth):
            layers.extend([
                EqualLinear(dim_latent, dim_latent, lr_mul),
                nn.LeakyReLU(leaky_relu_p)
            ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# simple MLP networks, but with latent variables
# the latent variables are the "genes" with the rest of the network as the scaffold for "gene expression" - as suggested in the paper

class MLP(Module):
    def __init__(
        self,
        dim,
        depth,
        dim_latent = 0,
        latent_mapping_network_depth = 2,
        expansion_factor = 2.
    ):
        super().__init__()
        dim_latent = default(dim_latent, 0)

        self.dim_latent = dim_latent

        self.needs_latent = dim_latent > 0

        self.encode_latent = nn.Sequential(
            LatentMappingNetwork(dim_latent, depth = latent_mapping_network_depth),
            Linear(dim_latent, dim * 2),
            nn.SiLU()
        ) if self.needs_latent else None

        dim_hidden = int(dim * expansion_factor)

        # layers

        layers = []

        for ind in range(depth):
            is_first = ind == 0

            film = None

            if self.needs_latent:
                film = FiLM(dim * 2, dim)

            lime = DynamicLIMe(dim, num_layers = ind + 1) if not is_first else None

            layer = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, dim),
            )

            layers.append(ModuleList([
                lime,
                film,
                layer
            ]))

        # modules across layers

        self.layers = ModuleList(layers)

        self.final_lime = DynamicLIMe(dim, depth + 1)

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

        # layers

        prev_layer_inputs = [x]

        for lime, film, layer in self.layers:

            layer_inp = x

            if exists(lime):
                layer_inp = lime(x, prev_layer_inputs)

            if exists(film):
                layer_inp = film(layer_inp, latent)

            x = layer(layer_inp) + x

            prev_layer_inputs.append(x)

        return self.final_lime(x, prev_layer_inputs)

# actor, critic, and agent (actor + critic)
# eventually, should just create a separate repo and aggregate all the MLP related architectures

class Actor(Module):
    def __init__(
        self,
        dim_state,
        num_actions,
        dim,
        mlp_depth,
        state_norm: StateNorm | None = None,
        dim_latent = 0,
    ):
        super().__init__()

        self.state_norm = state_norm

        self.dim_latent = dim_latent

        self.init_layer = nn.Sequential(
            nn.Linear(dim_state, dim),
            nn.SiLU()
        )

        self.mlp = MLP(dim = dim, depth = mlp_depth, dim_latent = dim_latent)

        self.to_out = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_actions, bias = False),
        )

    def forward(
        self,
        state,
        latent
    ):
        if exists(self.state_norm):
            with torch.no_grad():
                self.state_norm.eval()
                state = self.state_norm(state)

        hidden = self.init_layer(state)

        hidden = self.mlp(hidden, latent)

        return self.to_out(hidden)

class Critic(Module):
    def __init__(
        self,
        dim_state,
        dim,
        mlp_depth,
        dim_latent = 0,
        use_regression = False,
        state_norm: StateNorm | None = None,
        hl_gauss_loss_kwargs: dict = dict(
            min_value = -100.,
            max_value = 100.,
            num_bins = 200
        )
    ):
        super().__init__()

        self.state_norm = state_norm

        self.dim_latent = dim_latent

        self.init_layer = nn.Sequential(
            nn.Linear(dim_state, dim),
            nn.SiLU()
        )

        self.mlp = MLP(dim = dim, depth = mlp_depth, dim_latent = dim_latent)

        self.final_norm = nn.RMSNorm(dim)

        self.to_pred = HLGaussLayer(
            dim = dim,
            use_regression = use_regression,
            hl_gauss_loss = hl_gauss_loss_kwargs
        )

        self.use_regression = use_regression

        hl_gauss_loss = self.to_pred.hl_gauss_loss

        self.maybe_bins_to_value = hl_gauss_loss if not use_regression else identity
        self.loss_fn = hl_gauss_loss if not use_regression else F.mse_loss

    def forward_for_loss(
        self,
        state,
        latent,
        old_values,
        target,
        eps_clip = 0.4,
        use_improved = True
    ):

        if exists(self.state_norm):
            with torch.no_grad():
                self.state_norm.eval()
                state = self.state_norm(state)

        logits = self.forward(state, latent, return_logits = True)

        value = self.maybe_bins_to_value(logits)

        loss_fn = partial(self.loss_fn, reduction = 'none')

        if use_improved:
            clipped_target = target.clamp(-eps_clip, eps_clip)

            old_values_lo = old_values - eps_clip
            old_values_hi = old_values + eps_clip

            def is_between(lo, hi):
                return (lo < value) & (value < hi)

            clipped_loss = loss_fn(logits, clipped_target)
            loss = loss_fn(logits, target)

            value_loss = torch.where(
                is_between(target, old_values_lo) | is_between(old_values_hi, target),
                0.,
                torch.min(loss, clipped_loss)
            )
        else:
            clipped_value = old_values + (value - old_values).clamp(-eps_clip, eps_clip)

            loss = loss_fn(logits, target)
            clipped_loss = loss_fn(clipped_value, target)

            value_loss = torch.max(loss, clipped_loss)

        return value_loss.mean()

    def forward(
        self,
        state,
        latent,
        return_logits = False
    ):

        hidden = self.init_layer(state)

        hidden = self.mlp(hidden, latent)

        hidden = self.final_norm(hidden)

        pred_kwargs = dict(return_logits = return_logits) if not self.use_regression else dict()
        return self.to_pred(hidden, **pred_kwargs)

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
        num_islands = 1,                 # add the island strategy, which has been effectively used in a few recent works
        frozen_latents = True,
        crossover_random = True,         # random interp from parent1 to parent2 for crossover, set to `False` for averaging (0.5 constant value)
        l2norm_latent = False,           # whether to enforce latents on hypersphere,
        frac_tournaments = 0.25,         # fraction of genes to participate in tournament - the lower the value, the more chance a less fit gene could be selected
        frac_natural_selected = 0.25,    # number of least fit genes to remove from the pool
        frac_elitism = 0.1,              # frac of population to preserve from being noised
        frac_migrate = 0.1,              # frac of population, excluding elites, that migrate between islands randomly. will use a designated set migration pattern (since for some reason using random it seems to be worse for me)
        mutation_strength = 1.,          # factor to multiply to gaussian noise as mutation to latents
        fast_genetic_algorithm = False,
        fast_ga_values = torch.linspace(1, 5, 10),
        should_run_genetic_algorithm: Module | None = None, # eq (3) in paper
        default_should_run_ga_gamma = 1.5,
        migrate_every = 100,                 # how many steps before a migration between islands
        apply_genetic_algorithm_every = 2,   # how many steps before crossover + mutation happens for genes
        init_latent_fn: Callable | None = None
    ):
        super().__init__()
        assert num_latents > 1

        maybe_l2norm = l2norm if l2norm_latent else identity

        init_fn = default(init_latent_fn, torch.randn)

        latents = init_fn((num_latents, dim_latent))

        if l2norm_latent:
            latents = maybe_l2norm(latents, dim = -1)

        self.num_latents = num_latents
        self.frozen_latents = frozen_latents
        self.latents = nn.Parameter(latents, requires_grad = not frozen_latents)

        self.maybe_l2norm = maybe_l2norm

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

        assert self.num_tournament_participants >= 2

        self.crossover_random  = crossover_random

        self.mutation_strength = mutation_strength
        self.mutation_strength_sampler = PowerLawDist(fast_ga_values) if fast_genetic_algorithm else None

        self.num_elites = int(frac_elitism * latents_per_island)
        self.has_elites = self.num_elites > 0

        latents_without_elites = num_latents - self.num_elites
        self.num_migrate = int(frac_migrate * latents_without_elites)

        if not exists(should_run_genetic_algorithm):
            should_run_genetic_algorithm = ShouldRunGeneticAlgorithm(gamma = default_should_run_ga_gamma)

        self.should_run_genetic_algorithm = should_run_genetic_algorithm

        self.can_migrate = num_islands > 1

        self.migrate_every = migrate_every
        self.apply_genetic_algorithm_every = apply_genetic_algorithm_every

        self.register_buffer('step', tensor(1))

    def get_distance(self):
        # returns latent euclidean distance as proxy for diversity

        latents = rearrange(self.latents, '(i p) g -> i p g', i = self.num_islands)

        distance = torch.cdist(latents, latents)

        return distance

    def advance_step_(self):
        self.step.add_(1)

    def firefly_step(
        self,
        fitness,
        beta0 = 2.,           # exploitation factor, moving fireflies of low light intensity to high
        gamma = 1.,           # controls light intensity decay over distance - setting this to zero will make firefly equivalent to vanilla PSO
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

        # maybe fireflies on hypersphere

        fireflies = self.maybe_l2norm(fireflies)

        if not inplace:
            return fireflies

        self.latents.copy_(fireflies)

    @torch.no_grad()
    # non-gradient optimization, at least, not on the individual level (taken care of by rl component)
    def genetic_algorithm_step(
        self,
        fitness, # Float['p'],
        inplace = True,
        migrate = None # trigger a migration in the setting of multiple islands, the loop outside will need to have some `migrate_every` hyperparameter
    ):
        device = self.latents.device

        if not divisible_by(self.step.item(), self.apply_genetic_algorithm_every):
            self.advance_step_()
            return

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

        genes = rearrange(genes, '(i p) ... -> i p ...', i = islands)

        orig_genes = genes

        # 1. natural selection is simple in silico
        # you sort the population by the fitness and slice off the least fit end

        sorted_indices = fitness.sort(dim = -1).indices
        natural_selected_indices = sorted_indices[..., -self.num_natural_selected:]
        natural_select_gene_indices = repeat(natural_selected_indices, '... -> ... g', g = genes.shape[-1])

        genes, fitness = genes.gather(1, natural_select_gene_indices), fitness.gather(1, natural_selected_indices)

        # 2. for finding pairs of parents to replete gene pool, we will go with the popular tournament strategy

        tournament_shape = (islands, pop_size_per_island - self.num_natural_selected, self.num_natural_selected) # (island, num children needed, natural selected population to be bred)

        rand_tournament_gene_ids = batch_randperm(tournament_shape, device)[..., :tournament_participants]
        rand_tournament_gene_ids_for_gather = rearrange(rand_tournament_gene_ids, 'i p t -> i (p t)')

        participant_fitness = fitness.gather(1, rand_tournament_gene_ids_for_gather)
        participant_fitness = rearrange(participant_fitness, 'i (p t) -> i p t', t = tournament_participants)

        parent_indices_at_tournament = participant_fitness.topk(2, dim = -1).indices
        parent_gene_ids = rand_tournament_gene_ids.gather(-1, parent_indices_at_tournament)

        parent_gene_ids_for_gather = repeat(parent_gene_ids, 'i p parents -> i (p parents) g', g = genes.shape[-1])

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

        # 5. mutate with gaussian noise

        if exists(self.mutation_strength_sampler):
            mutation_strength = self.mutation_strength_sampler(genes.shape[:1])
        else:
            mutation_strength = self.mutation_strength

        genes = mutation(genes, mutation_strength = mutation_strength)

        # 6. maybe migration

        migrate = self.can_migrate and default(migrate, divisible_by(self.step.item(), self.migrate_every))

        if migrate:
            randperm = torch.randn(genes.shape[:-1], device = device).argsort(dim = -1)

            migrate_mask = randperm < self.num_migrate

            nonmigrants = rearrange(genes[~migrate_mask], '(i p) g -> i p g', i = islands)
            migrants = rearrange(genes[migrate_mask], '(i p) g -> i p g', i = islands)
            migrants = torch.roll(migrants, 1, dims = 0)

            genes = cat((nonmigrants, migrants), dim = 1)

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

        self.advance_step_()

    def forward(
        self,
        latent_id: int | None = None,
        *args,
        net: Module | None = None,
        net_latent_kwarg_name = 'latent',
        **kwargs,
    ):
        device = self.latents.device

        # if only 1 latent, assume doing ablation and get lone gene

        if not exists(latent_id) and self.num_latents == 1:
            latent_id = 0

        assert exists(latent_id)

        if not is_tensor(latent_id):
            latent_id = tensor(latent_id, device = device)

        assert (0 <= latent_id).all() and (latent_id < self.num_latents).all()

        # fetch latent

        latent = self.latents[latent_id]

        latent = self.maybe_l2norm(latent)

        if not exists(net):
            return latent

        latent_kwarg = {net_latent_kwarg_name: latent}

        return net(
            *args,
            **latent_kwarg,
            **kwargs
        )

# agent class

class Agent(Module):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        latent_gene_pool: LatentGenePool | None,
        optim_klass = AdoptAtan2,
        state_norm: StateNorm | None = None,
        actor_lr = 8e-4,
        critic_lr = 8e-4,
        latent_lr = 1e-5,
        actor_weight_decay = 5e-4,
        critic_weight_decay = 5e-4,
        diversity_aux_loss_weight = 0.,
        use_critic_ema = True,
        critic_ema_beta = 0.95,
        max_grad_norm = 1.0,
        batch_size = 32,
        calc_gae_kwargs: dict = dict(
            use_accelerated = False,
            gamma = 0.99,
            lam = 0.95,
        ),
        actor_loss_kwargs: dict = dict(
            eps_clip = 0.2,
            entropy_weight = .01,
            norm_advantages = True
        ),
        critic_loss_kwargs: dict = dict(
            eps_clip = 0.4
        ),
        use_improved_critic_loss = True,
        shrink_and_perturb_every = None,
        shrink_and_perturb_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        actor_optim_kwargs: dict = dict(),
        critic_optim_kwargs: dict = dict(),
        latent_optim_kwargs: dict = dict(),
        get_fitness_scores: Callable[..., Tensor] = get_fitness_scores,
        wrap_with_accelerate: bool = True,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        # hf accelerate

        self.wrap_with_accelerate = wrap_with_accelerate

        if wrap_with_accelerate:
            accelerate = Accelerator(**accelerate_kwargs)
            self.accelerate = accelerate

        # state norm

        self.state_norm = state_norm

        # actor, critic, and their shared latent gene pool

        self.actor = actor

        self.critic = critic

        if exists(state_norm):
            # insurance
            actor.state_norm = critic.state_norm = state_norm

        self.use_critic_ema = use_critic_ema

        self.critic_ema = EMA(
            critic,
            beta = critic_ema_beta,
            include_online_model = False,
            ignore_startswith_names = {'state_norm'},
            **ema_kwargs
        ) if use_critic_ema else None

        self.latent_gene_pool = latent_gene_pool
        self.num_latents = latent_gene_pool.num_latents if exists(latent_gene_pool) else 1
        self.has_latent_genes = exists(latent_gene_pool)

        assert actor.dim_latent == critic.dim_latent

        if self.has_latent_genes:
            assert latent_gene_pool.dim_latent == actor.dim_latent

        # gae function

        self.calc_gae = partial(calc_generalized_advantage_estimate, **calc_gae_kwargs)

        # actor critic loss related

        self.actor_loss = partial(actor_loss, **actor_loss_kwargs)
        self.critic_loss_kwargs = critic_loss_kwargs

        self.use_improved_critic_loss = use_improved_critic_loss

        # fitness score related

        self.get_fitness_scores = get_fitness_scores

        # learning hparams

        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.has_grad_clip = exists(max_grad_norm)

        # optimizers

        self.actor_optim = optim_klass(actor.parameters(), lr = actor_lr, weight_decay = actor_weight_decay, **actor_optim_kwargs)
        self.critic_optim = optim_klass(critic.parameters(), lr = critic_lr, weight_decay = critic_weight_decay, **critic_optim_kwargs)

        self.latent_optim = optim_klass(latent_gene_pool.parameters(), lr = latent_lr, **latent_optim_kwargs) if exists(latent_gene_pool) and not latent_gene_pool.frozen_latents else None

        # shrink and perturb every

        self.should_noise_weights = exists(shrink_and_perturb_every)
        self.shrink_and_perturb_every = shrink_and_perturb_every
        self.shrink_and_perturb_ = partial(shrink_and_perturb_, **shrink_and_perturb_kwargs)

        # promotes latents to be farther apart for diversity maintenance

        self.has_diversity_loss = diversity_aux_loss_weight > 0.
        self.diversity_aux_loss_weight = diversity_aux_loss_weight

        # wrap with accelerate

        self.unwrap_model = identity if not wrap_with_accelerate else self.accelerate.unwrap_model

        step = tensor(0)

        self.clip_grad_norm_ = nn.utils.clip_grad_norm_

        if wrap_with_accelerate:
            self.clip_grad_norm_ = self.accelerate.clip_grad_norm_

            (
                self.state_norm,
                self.actor,
                self.critic,
                self.latent_gene_pool,
                self.actor_optim,
                self.critic_optim,
                self.latent_optim,
            ) = tuple(
                maybe(self.accelerate.prepare)(m) for m in (
                    self.state_norm,
                    self.actor,
                    self.critic,
                    self.latent_gene_pool,
                    self.actor_optim,
                    self.critic_optim,
                    self.latent_optim,
                )
            )

            if exists(self.critic_ema):
                self.critic_ema.to(self.accelerate.device)

            step = step.to(self.accelerate.device)

        # device tracking

        self.register_buffer('step', step)

    @property
    def device(self):
        return self.step.device

    @property
    def unwrapped_latent_gene_pool(self):
        return self.unwrap_model(self.latent_gene_pool)

    def log(self, **data_kwargs):
        if not self.wrap_with_accelerate:
            return

        self.accelerate.log(data_kwargs, step = self.step)

    def save(self, path, overwrite = False):
        path = Path(path)
        unwrap = self.unwrap_model

        assert not path.exists() or overwrite

        pkg = dict(
            state_norm = unwrap(self.state_norm).state_dict() if self.state_norm else None,
            actor = unwrap(self.actor).state_dict(),
            critic = unwrap(self.critic).state_dict(),
            critic_ema = self.critic_ema.state_dict() if self.use_critic_ema else None,
            latents = unwrap(self.latent_gene_pool).state_dict() if self.has_latent_genes else None,
            actor_optim = unwrap(self.actor_optim).state_dict(),
            critic_optim = unwrap(self.critic_optim).state_dict(),
            latent_optim = unwrap(self.latent_optim).state_dict() if exists(self.latent_optim) else None
        )

        torch.save(pkg, str(path))

    def load(self, path):
        unwrap = self.unwrap_model
        path = Path(path)

        assert path.exists()

        pkg = torch.load(str(path), weights_only = True)

        unwrap(self.actor).load_state_dict(pkg['actor'])

        unwrap(self.critic).load_state_dict(pkg['critic'])

        if self.use_critic_ema:
            self.critic_ema.load_state_dict(pkg['critic_ema'])

        if exists(pkg.get('latents', None)):
            self.latent_gene_pool.load_state_dict(pkg['latents'])

        unwrap(self.actor_optim).load_state_dict(pkg['actor_optim'])
        unwrap(self.critic_optim).load_state_dict(pkg['critic_optim'])

        if exists(pkg.get('latent_optim', None)):
            unwrap(self.latent_optim).load_state_dict(pkg['latent_optim'])

    @move_input_tensors_to_device
    def get_actor_actions(
        self,
        state,
        latent_id = None,
        latent = None,
        sample = False,
        temperature = 1.,
        use_unwrapped_model = False
    ):
        maybe_unwrap = identity if not use_unwrapped_model else self.unwrap_model

        if not exists(latent) and exists(latent_id):
            latent = maybe_unwrap(self.latent_gene_pool)(latent_id = latent_id)

        logits = maybe_unwrap(self.actor)(state, latent)

        if not sample:
            return logits

        actions = gumbel_sample(logits, temperature = temperature)

        log_probs = gather_log_prob(logits, actions)

        return actions, log_probs

    @move_input_tensors_to_device
    def get_critic_values(
        self,
        state,
        latent_id = None,
        latent = None,
        use_ema_if_available = False,
        use_unwrapped_model = False
    ):

        maybe_unwrap = identity if not use_unwrapped_model else self.unwrap_model

        if not exists(latent) and exists(latent_id):
            latent = maybe_unwrap(self.latent_gene_pool)(latent_id = latent_id)

        critic_forward = maybe_unwrap(self.critic)

        if use_ema_if_available and self.use_critic_ema:
            critic_forward = self.critic_ema

        return critic_forward(state, latent)

    def update_latent_gene_pool_(
        self,
        fitnesses
    ):
        if not self.has_latent_genes:
            return

        return self.latent_gene_pool.genetic_algorithm_step(fitnesses)

    def learn_from(
        self,
        memories_and_cumulative_rewards: MemoriesAndCumulativeRewards,
        epochs = 2

    ):
        memories_and_cumulative_rewards = to_device(memories_and_cumulative_rewards, self.device)

        memories_list, rewards_per_latent_episode = memories_and_cumulative_rewards

        # stack memories

        memories = map(stack, zip(*memories_list))

        memories_list.clear()

        maybe_barrier()

        if is_distributed():
            memories = map(partial(all_gather_variable_dim, dim = 0), memories)

            rewards_per_latent_episode = dist.all_reduce(rewards_per_latent_episode)

        # calculate fitness scores

        fitness_scores = self.get_fitness_scores(rewards_per_latent_episode, memories)

        # process memories

        (
            episode_ids,
            states,
            latent_gene_ids,
            actions,
            log_probs,
            rewards,
            values,
            dones
        ) = memories

        masks = 1. - dones.float()

        # generalized advantage estimate

        advantages = self.calc_gae(
            rewards,
            values,
            masks,
        )

        # dataset and dataloader

        valid_episode = episode_ids >= 0

        dataset = TensorDataset(*[t[valid_episode] for t in (advantages, states, latent_gene_ids, actions, log_probs, values)])

        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        if self.wrap_with_accelerate:
            dataloader = self.accelerate.prepare(dataloader)

        # updating actor and critic

        self.actor.train()
        self.critic.train()

        for _ in tqdm(range(epochs), desc = 'learning actor/critic epoch'):
            for (
                advantages,
                states,
                latent_gene_ids,
                actions,
                log_probs,
                old_values
            ) in dataloader:

                if self.has_latent_genes:
                    latents = self.latent_gene_pool(latent_id = latent_gene_ids)

                    orig_latents = latents
                    latents = latents.detach()
                    latents.requires_grad_()
                else:
                    latents = None

                # learn actor

                logits = self.actor(states, latents)

                actor_loss = self.actor_loss(logits, log_probs, actions, advantages)

                actor_loss.backward()

                if exists(self.has_grad_clip):
                    self.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                self.actor_optim.step()
                self.actor_optim.zero_grad()

                # learn critic with maybe classification loss

                critic_loss = self.critic.forward_for_loss(
                    states,
                    latents,
                    old_values = old_values,
                    target = advantages + old_values,
                    use_improved = self.use_improved_critic_loss,
                    **self.critic_loss_kwargs
                )

                critic_loss.backward()

                if exists(self.has_grad_clip):
                    self.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.critic_optim.step()
                self.critic_optim.zero_grad()

                # log actor critic loss

                self.log(
                    actor_loss = actor_loss.item(),
                    critic_loss = critic_loss.item(),
                    fitness_scores = fitness_scores
                )

                # maybe ema update critic

                if self.use_critic_ema:
                    self.critic_ema.update()

                # maybe update latents, if not frozen

                if not self.has_latent_genes or self.latent_gene_pool.frozen_latents:
                    continue

                orig_latents.backward(latents.grad)

                if self.has_diversity_loss:
                    diversity = self.latent_gene_pool.get_distance()
                    diversity_loss = (-diversity).tril(-1).exp().mean()

                    (diversity_loss * self.diversity_aux_loss_weight).backward()

                if exists(self.has_grad_clip):
                    self.clip_grad_norm_(self.latent_gene_pool.parameters(), self.max_grad_norm)

                self.latent_optim.step()
                self.latent_optim.zero_grad()

                if self.has_diversity_loss:
                    self.log(
                        diversity_loss = diversity_loss.item()
                    )

        # update state norm if needed

        if exists(self.state_norm):
            self.state_norm.train()

            for _, states, *_ in tqdm(dataloader, desc = 'state norm learning'):
                self.state_norm(states)

        # apply evolution

        if self.has_latent_genes:
            self.latent_gene_pool.genetic_algorithm_step(fitness_scores)

        # maybe shrink and perturb

        if self.should_noise_weights and divisible_by(self.step.item(), self.shrink_and_perturb_every):
            self.shrink_and_perturb_(self.actor)
            self.shrink_and_perturb_(self.critic)

        # increment step

        self.step.add_(1)

# reinforcement learning related - ppo

def actor_loss(
    logits,         # Float[b l]
    old_log_probs,  # Float[b]
    actions,        # Int[b]
    advantages,     # Float[b]
    eps_clip = 0.2,
    entropy_weight = .01,
    eps = 1e-5,
    norm_advantages = True
):
    batch = logits.shape[0]

    log_probs = gather_log_prob(logits, actions)

    ratio = (log_probs - old_log_probs).exp()

    # classic clipped surrogate loss from ppo

    clipped_ratio = ratio.clamp(min = 1. - eps_clip, max = 1. + eps_clip)

    if norm_advantages:
        advantages = F.layer_norm(advantages, (batch,), eps = eps)

    actor_loss = -torch.min(clipped_ratio * advantages, ratio * advantages)

    # add entropy loss for exploration

    entropy = calc_entropy(logits)

    entropy_aux_loss = -entropy_weight * entropy

    return (actor_loss + entropy_aux_loss).mean()

# agent contains the actor, critic, and the latent genetic pool

def create_agent(
    *,
    dim_state,
    num_latents,
    dim_latent,
    actor_num_actions,
    actor_dim,
    actor_mlp_depth,
    critic_dim,
    critic_mlp_depth,
    use_critic_ema = True,
    latent_gene_pool_kwargs: dict = dict(),
    actor_kwargs: dict = dict(),
    critic_kwargs: dict = dict(),
    **kwargs
) -> Agent:

    has_latent_genes = num_latents > 1

    if not has_latent_genes:
        dim_latent = None

    latent_gene_pool = LatentGenePool(
        num_latents = num_latents,
        dim_latent = dim_latent,
        **latent_gene_pool_kwargs
    ) if has_latent_genes else None

    state_norm = StateNorm(dim = dim_state)

    actor = Actor(
        num_actions = actor_num_actions,
        dim_state = dim_state,
        dim_latent = dim_latent,
        dim = actor_dim,
        mlp_depth = actor_mlp_depth,
        state_norm = state_norm,
        **actor_kwargs
    )

    critic = Critic(
        dim_state = dim_state,
        dim_latent = dim_latent,
        dim = critic_dim,
        mlp_depth = critic_mlp_depth,
        state_norm = state_norm,
        **critic_kwargs
    )

    agent = Agent(
        actor = actor,
        critic = critic,
        state_norm = state_norm,
        latent_gene_pool = latent_gene_pool,
        use_critic_ema = use_critic_ema,
        **kwargs
    )

    return agent

# EPO - which is just PPO with natural selection of a population of latent variables conditioning the agent
# the tricky part is that the latent ids for each episode / trajectory needs to be tracked

Memory = namedtuple('Memory', [
    'episode_id',
    'state',
    'latent_gene_id',
    'action',
    'log_prob',
    'reward',
    'value',
    'done'
])

MemoriesAndCumulativeRewards = namedtuple('MemoriesAndCumulativeRewards', [
    'memories',
    'cumulative_rewards' # Float['latent episodes']
])

class EPO(Module):

    def __init__(
        self,
        agent: Agent,
        episodes_per_latent,
        max_episode_length,
        action_sample_temperature = 1.,
        fix_environ_across_latents = True
    ):
        super().__init__()
        self.agent = agent
        self.action_sample_temperature = action_sample_temperature

        self.num_latents = agent.latent_gene_pool.num_latents if agent.has_latent_genes else 1
        self.episodes_per_latent = episodes_per_latent
        self.max_episode_length = max_episode_length
        self.fix_environ_across_latents = fix_environ_across_latents

        self.register_buffer('dummy', tensor(0, device = agent.device))

    @property
    def device(self):
        return self.dummy.device

    def rollouts_for_machine(
        self,
        fix_environ_across_latents = False
    ): # -> (<latent_id>, <episode_id>, <maybe synced env seed>) for the machine

        num_latents = self.num_latents
        episodes = self.episodes_per_latent
        num_latent_episodes = num_latents * episodes

        # if fixing environment across latents, compute all the environment seeds upfront for simplicity

        environment_seeds = None

        if fix_environ_across_latents:
            environment_seeds = torch.randint(0, int(1e6), (episodes,))

            if is_distributed():
                dist.all_reduce(environment_seeds) # reduce sum as a way to synchronize. it's fine

        # get number of machines, and this machine id

        world_size, rank = get_world_and_rank()

        assert num_latent_episodes >= world_size, f'number of ({self.num_latents} latents x {self.episodes_per_latent} episodes) ({num_latent_episodes}) must be greater than world size ({world_size}) for now'

        latent_episode_permutations = list(product(range(num_latents), range(episodes)))

        num_rollouts_per_machine = ceil(num_latent_episodes / world_size)

        for i in range(num_rollouts_per_machine):
            rollout_id = rank * num_rollouts_per_machine + i

            if rollout_id >= num_latent_episodes:
                continue

            latent_id, episode_id = latent_episode_permutations[rollout_id]

            # maybe synchronized environment seed

            maybe_seed = None
            if fix_environ_across_latents:
                maybe_seed = environment_seeds[episode_id]

            yield latent_id, episode_id, maybe_seed.item()

    @torch.no_grad()
    def gather_experience_from(
        self,
        env,
        memories: list[Memory] | None = None,
        fix_environ_across_latents = None
    ) -> MemoriesAndCumulativeRewards:

        fix_environ_across_latents = default(fix_environ_across_latents, self.fix_environ_across_latents)

        self.agent.eval()

        invalid_episode = tensor(-1) # will use `episode_id` value of `-1` for the `next_value`, needed for not discarding last reward for generalized advantage estimate

        if not exists(memories):
            memories = []

        rewards_per_latent_episode = torch.zeros((self.num_latents, self.episodes_per_latent), device = self.device)

        rollout_gen = self.rollouts_for_machine(fix_environ_across_latents)

        for latent_id, episode_id, maybe_seed in tqdm(rollout_gen, desc = 'rollout'):

            time = 0

            # initial state

            reset_kwargs = dict()

            if fix_environ_across_latents:
                reset_kwargs.update(seed = maybe_seed)

            state, _ = interface_torch_numpy(env.reset, device = self.device)(**reset_kwargs)

            # get latent from pool

            latent = self.agent.unwrapped_latent_gene_pool(latent_id = latent_id) if self.agent.has_latent_genes else None

            # until maximum episode length

            done = tensor(False)

            while time < self.max_episode_length and not done:

                # sample action

                action, log_prob = temp_batch_dim(self.agent.get_actor_actions)(state, latent = latent, sample = True, temperature = self.action_sample_temperature, use_unwrapped_model = True)

                # values

                value = temp_batch_dim(self.agent.get_critic_values)(state, latent = latent, use_ema_if_available = True, use_unwrapped_model = True)

                # get the next state, action, and reward

                state, reward, truncated, terminated, _ = interface_torch_numpy(env.step, device = self.device)(action)

                done = truncated or terminated

                # update cumulative rewards per latent, to be used as default fitness score

                rewards_per_latent_episode[latent_id, episode_id] += reward

                # store memories

                memory = Memory(
                    tensor(episode_id),
                    state,
                    tensor(latent_id),
                    action,
                    log_prob,
                    reward,
                    value,
                    terminated
                )

                memory = Memory(*tuple(t.cpu() for t in memory))

                memories.append(memory)

                time += 1

            if not terminated:
                # add bootstrap value if truncated

                next_value = temp_batch_dim(self.agent.get_critic_values)(state, latent = latent, use_ema_if_available = True, use_unwrapped_model = True)

                memory_for_gae = memory._replace(
                    episode_id = invalid_episode,
                    value = next_value.cpu(),
                    done = tensor(True)
                )

                memories.append(memory_for_gae)

        return MemoriesAndCumulativeRewards(
            memories = memories,
            cumulative_rewards = rewards_per_latent_episode
        )

    def forward(
        self,
        agent: Agent,
        env,
        num_learning_cycles,
        seed = None
    ):

        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

        for _ in tqdm(range(num_learning_cycles), desc = 'learning cycle'):

            memories = self.gather_experience_from(env)

            agent.learn_from(memories)

        print('training complete')
