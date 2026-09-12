from __future__ import annotations

import math
from collections import namedtuple
from copy import deepcopy
from functools import partial, wraps
from itertools import product
from math import ceil
from pathlib import Path
from typing import Callable

import einx
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from assoc_scan import AssocScan
from einops import einsum, rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from hl_gauss_pytorch import HLGaussLayer
from mean_conc_beta import Beta
from torch import Tensor, cat, from_numpy, is_tensor, nn, stack, tensor
from torch.distributions import Categorical, Distribution
from torch.nn import Linear, Module, ModuleList
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from x_mlps_pytorch import AttnResidualNormedMLP

from evolutionary_policy_optimization.distributed import all_gather, get_world_and_rank, is_distributed, maybe_barrier, maybe_sync_seed

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t, *args, **kwargs):
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
        if isinstance(t, np.ndarray):
            t = from_numpy(np.array(t))
        elif isinstance(t, np.generic):
            t = tensor(t.item())
        elif isinstance(t, (float, int, bool)):
            t = tensor(t)

        if is_tensor(t) and t.is_floating_point():
            t = t.float()

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

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

def batch_randperm(shape, device):
    return torch.randn(shape, device = device).argsort(dim = -1)

def sum_to_batch(t):
    # fold trailing action dims into one value per state

    return reduce(t, 'b ... -> b', 'sum')

def mode_action(distr):
    # greedy action - argmax for categorical, mean for beta

    if isinstance(distr, Categorical):
        return distr.probs.argmax(dim = -1)

    return distr.mean

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

    device = next(module.parameters()).device
    maybe_sync_seed(device)

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
        assert state.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {state.shape[-1]}'

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
# https://arxiv.org/abs/1912.04958

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

            assert latent.shape[0] == x.shape[0], f'received state with batch size {x.shape[0]} but latent ids received had batch size {latent.shape[0]}'

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

# discriminator for predicting latent code from states and actions (DIAYN - Eysenbach et al. 2018)

class DiversityDiscr(Module):
    def __init__(
        self,
        dim_state,
        num_latents,
        dim = 64,
        depth = 2
    ):
        super().__init__()
        self.state_proj = nn.Linear(dim_state, dim)

        self.net = AttnResidualNormedMLP(
            dim = dim,
            depth = depth,
            dim_in = dim * 2,
            dim_out = num_latents
        )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)):
                module.reset_parameters()

    def forward(self, state, next_state):
        state_embed = self.state_proj(state)
        next_state_embed = self.state_proj(next_state)
        return self.net((state_embed, next_state_embed))

# action distributions - the actor always returns a `Distribution`, either
# categorical (discrete) or beta mean-conc (continuous). beta is defined over
# (-1, 1) - multiplied by some constant usually (e.g. 0.4) for the env action range

class CategoricalActionDistr(Module):
    def forward(self, logits, temperature = 1.):
        if temperature > 0. and temperature != 1.:
            logits = logits / temperature

        return Categorical(logits = logits)

# self-predictive representations (SPR) over the actor's hidden embedding

class HiddenSpr(Module):
    def __init__(
        self,
        actor,
        dim_hidden,
        num_actions,
        dim_action = 32,
    ):
        super().__init__()
        action_is_continuous = actor.beta_actions
        self.action_is_continuous = action_is_continuous

        if action_is_continuous:
            self.action_proj = nn.Linear(num_actions, dim_action)
        else:
            self.action_proj = nn.Embedding(num_actions, dim_action)

        self.to_dynamics = nn.Sequential(
            nn.Linear(dim_hidden + dim_action, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )

        self.proj_head = nn.Linear(dim_hidden, dim_hidden, bias = False)

        # ema target of the actor body + projection head

        self.target_actor = deepcopy(actor).requires_grad_(False)
        self.target_proj = deepcopy(self.proj_head).requires_grad_(False)

    def online_parameters(self):
        return [
            *self.action_proj.parameters(),
            *self.to_dynamics.parameters(),
            *self.proj_head.parameters(),
        ]

    def predict(self, hidden, action):
        if not self.action_is_continuous:
            action = action.long()
            if action.ndim > 1:
                action = rearrange(action, '... 1 -> ...')

        action_proj = self.action_proj(action)
        dynamics = self.to_dynamics(cat((hidden, action_proj), dim = -1))
        return self.proj_head(dynamics + hidden)

    @torch.no_grad()
    def target(self, state, latent):
        return self.target_proj(self.target_actor.latent(state, latent))

    @torch.no_grad()
    def _ema_update_one_(self, target_module, source_module, decay):
        # if decay is 0, this is a hard copy of the source into the target

        for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
            target_param.lerp_(source_param, 1. - decay)

    @torch.no_grad()
    def ema_update(self, actor, decay):
        self._ema_update_one_(self.target_actor.init_layer, actor.init_layer, decay)
        self._ema_update_one_(self.target_actor.mlp, actor.mlp, decay)
        self._ema_update_one_(self.target_proj, self.proj_head, decay)

    @torch.no_grad()
    def sync_targets_(self, actor):
        self.ema_update(actor, decay = 0.)

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
        action_is_continuous = False, # continuous control - beta policy
        beta_kwargs: dict = dict(),
    ):
        super().__init__()

        self.state_norm = state_norm

        self.dim = dim
        self.num_actions = num_actions
        self.dim_latent = dim_latent
        self.beta_actions = action_is_continuous

        self.init_layer = nn.Sequential(
            nn.Linear(dim_state, dim),
            nn.SiLU()
        )

        self.mlp = MLP(dim = dim, depth = mlp_depth, dim_latent = dim_latent)

        if self.beta_actions:
            # beta head - (raw mean, raw concentration) per action dim

            self.to_out = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, num_actions * 2, bias = False),
                Rearrange('... (d params) -> ... d params', params = 2),
            )
        else:
            self.to_out = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, num_actions, bias = False),
            )

        self.action_distr = Beta(**beta_kwargs) if self.beta_actions else CategoricalActionDistr()

    def latent(
        self,
        state,
        latent
    ) -> Tensor:
        # hidden embedding right before the action projection -
        # the dynamics in hidden_spr operate over this

        if exists(self.state_norm):
            with torch.no_grad():
                self.state_norm.eval()
                state = self.state_norm(state)

        hidden = self.init_layer(state)

        return self.mlp(hidden, latent)

    def forward(
        self,
        state,
        latent,
        temperature = 1.
    ) -> Distribution:
        hidden = self.latent(state, latent)
        return self.action_distr(self.to_out(hidden), temperature = temperature)

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
            min_value = 0.,
            max_value = 500.,
            num_bins = 250
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
        target,
        old_values = None,
        clip_value = False,
        eps_clip = 0.8,
        use_improved = True
    ):

        if exists(self.state_norm):
            with torch.no_grad():
                self.state_norm.eval()
                state = self.state_norm(state)

        logits = self.forward(state, latent, return_logits = True)

        if not clip_value or not exists(old_values) or not exists(eps_clip):
            return self.loss_fn(logits, target)

        value = self.maybe_bins_to_value(logits)

        loss_fn = partial(self.loss_fn, reduction = 'none')

        if use_improved:
            old_values_lo = old_values - eps_clip
            old_values_hi = old_values + eps_clip

            clipped_target = target.clamp(old_values_lo, old_values_hi)

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

        if self.use_regression:
            return self.to_pred(hidden)

        logits = self.to_pred(hidden, return_logits = True)

        if return_logits:
            return logits

        value = self.maybe_bins_to_value(logits)

        return value

# criteria for running genetic algorithm

class ShouldRunGeneticAlgorithm(Module):
    def __init__(
        self,
        gamma = 0.25,     # fire when the spread exceeds this fraction of the fitness level
        min_spread = 1.0, # absolute spread floor
    ):
        super().__init__()
        self.gamma = gamma
        self.min_spread = min_spread

    def forward(self, fitnesses):
        # eq (3) - fire when the fitness spread is a meaningful fraction of the level

        spread = fitnesses.amax(dim = -1) - fitnesses.amin(dim = -1)
        scale = torch.abs(fitnesses.median(dim = -1).values)

        return spread > (self.gamma * scale + self.min_spread)

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
        default_should_run_ga_gamma = 0.25,
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

        maybe_sync_seed(device)

        if not divisible_by(self.step.item(), self.apply_genetic_algorithm_every):
            self.advance_step_()

            if inplace:
                return False, None

            return False, self.latents

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
            self.advance_step_()

            if inplace:
                return False, None

            return False, genes

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

        ret = (True, genes)

        if not inplace:
            return ret

        # store the genes for the next interaction with environment for new fitness values (a function of reward and other to be researched measures)

        self.latents.copy_(genes)

        self.advance_step_()

        if inplace:
            return ret

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
        optim_klass = AdamW,
        state_norm: StateNorm | None = None,
        actor_lr = 8e-4,
        critic_lr = 8e-4,
        latent_lr = 1e-5,
        actor_weight_decay = 0.,
        critic_weight_decay = 0.,
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
        clip_value = False,
        critic_loss_kwargs: dict = dict(
            eps_clip = 0.8
        ),
        use_spo = False, # Simple Policy Optimization - Xie et al. https://arxiv.org/abs/2401.16025v9
        use_improved_critic_loss = True,
        shrink_and_perturb_every = None,
        shrink_and_perturb_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        actor_optim_kwargs: dict = dict(),
        critic_optim_kwargs: dict = dict(),
        latent_optim_kwargs: dict = dict(),
        use_diversity_discr = False,
        diversity_discr_kwargs: dict = dict(dim = 64, depth = 2),
        diversity_discr_lr = 3e-4,
        diversity_discr_optim_kwargs: dict = dict(),
        use_hidden_spr = False, # self-predictive representations over the actor's hidden embedding - https://arxiv.org/abs/2106.04799
        hidden_spr_dim_action = 32,
        hidden_spr_lr = 3e-4,
        hidden_spr_weight = 1.0,
        hidden_spr_ema_update = 0.99,
        get_fitness_scores: Callable[..., Tensor] = get_fitness_scores,
        wrap_with_accelerate: bool = True,
        accelerate_kwargs: dict = dict(),
        accelerator = None,
        quiet: bool = False,
    ):
        super().__init__()

        self.quiet = quiet

        # hf accelerate

        self.wrap_with_accelerate = wrap_with_accelerate

        if wrap_with_accelerate:
            self.accelerate = accelerator if exists(accelerator) else Accelerator(**accelerate_kwargs)

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

        self.use_spo = use_spo
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

        # diversity discriminator

        self.use_diversity_discr = use_diversity_discr

        if use_diversity_discr:
            assert exists(latent_gene_pool), 'latent_gene_pool must be present to use DIAYN'
            dim_state = actor.init_layer[0].in_features

            self.diversity_discr = DiversityDiscr(
                dim_state = dim_state,
                num_latents = latent_gene_pool.num_latents,
                **diversity_discr_kwargs
            )

            self.diversity_discr_optim = optim_klass(self.diversity_discr.parameters(), lr = diversity_discr_lr, **diversity_discr_optim_kwargs)
        else:
            self.diversity_discr = None
            self.diversity_discr_optim = None

        self.clip_value = clip_value
        self.critic_loss_kwargs = critic_loss_kwargs

        # self-predictive representations (SPR)

        self.use_hidden_spr = use_hidden_spr
        self.hidden_spr_weight = hidden_spr_weight

        if use_hidden_spr:
            self.hidden_spr = HiddenSpr(
                actor,
                dim_hidden = actor.dim,
                num_actions = actor.num_actions,
                dim_action = hidden_spr_dim_action,
            )

            self.hidden_spr_optim = optim_klass(self.hidden_spr.online_parameters(), lr = hidden_spr_lr)
            self.hidden_spr_ema_update = hidden_spr_ema_update
        else:
            self.hidden_spr = None
            self.hidden_spr_optim = None

        self.register_buffer('has_diversity_discr_warmed_up', tensor(False))
        self.register_buffer('zero', tensor(0.))

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
            device = self.accelerate.device

            # device placement for modules without gradient parameters

            for m in (self.state_norm, self.latent_gene_pool):
                if exists(m):
                    m.to(device)

            # DDP wrap models with gradient parameters + prepare optimizers

            (
                self.actor,
                self.critic,
                self.actor_optim,
                self.critic_optim,
            ) = self.accelerate.prepare(
                self.actor,
                self.critic,
                self.actor_optim,
                self.critic_optim,
            )

            if exists(self.diversity_discr):
                self.diversity_discr, self.diversity_discr_optim = self.accelerate.prepare(
                    self.diversity_discr, self.diversity_discr_optim
                )

            if exists(self.hidden_spr):
                self.hidden_spr, self.hidden_spr_optim = self.accelerate.prepare(
                    self.hidden_spr, self.hidden_spr_optim
                )

            if exists(self.latent_optim):
                self.latent_optim = self.accelerate.prepare(self.latent_optim)

            if exists(self.critic_ema):
                self.critic_ema.to(device)

            step = step.to(device)

        # device tracking

        self.register_buffer('step', step)

    @property
    def device(self):
        return self.step.device

    @property
    def is_main_process(self):
        return not self.wrap_with_accelerate or self.accelerate.is_main_process

    @property
    def unwrapped_latent_gene_pool(self):
        return self.unwrap_model(self.latent_gene_pool)

    def log(self, **data_kwargs):
        if not self.wrap_with_accelerate:
            return

        self.accelerate.log(data_kwargs, step = self.step)

    def save(self, path, overwrite = False):
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        unwrap = self.unwrap_model
        unwrap_optim = lambda opt: opt.optimizer if hasattr(opt, 'optimizer') else opt

        assert not path.exists() or overwrite

        pkg = dict(
            actor = unwrap(self.actor).state_dict(),
            critic = unwrap(self.critic).state_dict(),
            critic_ema = self.critic_ema.state_dict() if self.use_critic_ema else None,
            latents = unwrap(self.latent_gene_pool).state_dict() if self.has_latent_genes else None,
            diversity_discr = unwrap(self.diversity_discr).state_dict() if self.use_diversity_discr else None,
            has_diversity_discr_warmed_up = self.has_diversity_discr_warmed_up.item(),
            actor_optim = unwrap_optim(self.actor_optim).state_dict(),
            critic_optim = unwrap_optim(self.critic_optim).state_dict(),
            latent_optim = unwrap_optim(self.latent_optim).state_dict() if exists(self.latent_optim) else None,
            diversity_discr_optim = unwrap_optim(self.diversity_discr_optim).state_dict() if self.use_diversity_discr else None,
            hidden_spr = unwrap(self.hidden_spr).state_dict() if self.use_hidden_spr else None,
            hidden_spr_optim = unwrap_optim(self.hidden_spr_optim).state_dict() if self.use_hidden_spr else None,
        )

        torch.save(pkg, str(path))

    def load(self, path):
        unwrap = self.unwrap_model
        unwrap_optim = lambda opt: opt.optimizer if hasattr(opt, 'optimizer') else opt
        path = Path(path)

        assert path.exists()

        pkg = torch.load(str(path), weights_only = True)

        unwrap(self.actor).load_state_dict(pkg['actor'])
        unwrap(self.critic).load_state_dict(pkg['critic'])

        if self.use_critic_ema:
            self.critic_ema.load_state_dict(pkg['critic_ema'])

        if 'latents' in pkg and exists(pkg['latents']):
            unwrap(self.latent_gene_pool).load_state_dict(pkg['latents'])

        if self.use_diversity_discr and 'diversity_discr' in pkg and exists(pkg['diversity_discr']):
            unwrap(self.diversity_discr).load_state_dict(pkg['diversity_discr'])

        if 'has_diversity_discr_warmed_up' in pkg:
            self.has_diversity_discr_warmed_up.copy_(tensor(pkg['has_diversity_discr_warmed_up']))

        if self.use_hidden_spr and 'hidden_spr' in pkg and exists(pkg['hidden_spr']):
            unwrap(self.hidden_spr).load_state_dict(pkg['hidden_spr'])
        elif self.use_hidden_spr:
            unwrap(self.hidden_spr).sync_targets_(unwrap(self.actor))

        unwrap_optim(self.actor_optim).load_state_dict(pkg['actor_optim'])
        unwrap_optim(self.critic_optim).load_state_dict(pkg['critic_optim'])

        if 'latent_optim' in pkg and exists(pkg['latent_optim']):
            unwrap_optim(self.latent_optim).load_state_dict(pkg['latent_optim'])

        if self.use_diversity_discr and 'diversity_discr_optim' in pkg and exists(pkg['diversity_discr_optim']):
            unwrap_optim(self.diversity_discr_optim).load_state_dict(pkg['diversity_discr_optim'])

        if self.use_hidden_spr and 'hidden_spr_optim' in pkg and exists(pkg['hidden_spr_optim']):
            unwrap_optim(self.hidden_spr_optim).load_state_dict(pkg['hidden_spr_optim'])

    @move_input_tensors_to_device
    def get_actor_distribution(
        self,
        state,
        latent_id = None,
        latent = None,
        temperature = 1.,
        use_unwrapped_model = False
    ) -> Distribution:
        maybe_unwrap = identity if not use_unwrapped_model else self.unwrap_model

        if not exists(latent) and exists(latent_id) and exists(self.latent_gene_pool):
            latent = maybe_unwrap(self.latent_gene_pool)(latent_id = latent_id)

        return maybe_unwrap(self.actor)(state, latent, temperature = temperature)

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
        distr = self.get_actor_distribution(
            state,
            latent_id = latent_id,
            latent = latent,
            temperature = temperature if sample else 1.,
            use_unwrapped_model = use_unwrapped_model
        )

        if not sample:
            return mode_action(distr)

        # temperature <= 0 is greedy - mode works for both discrete and continuous

        actions = mode_action(distr) if temperature <= 0. else distr.sample()

        log_probs = sum_to_batch(distr.log_prob(actions))

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

        if not exists(latent) and exists(latent_id) and exists(self.latent_gene_pool):
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
            memories = [all_gather(m) for m in memories]
            dist.all_reduce(rewards_per_latent_episode)

        # calculate fitness scores

        fitness_scores = self.get_fitness_scores(rewards_per_latent_episode, memories)

        # process memories

        (
            episode_ids,
            states,
            next_states,
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

        dataset = TensorDataset(*[t[valid_episode] for t in (advantages, states, next_states, latent_gene_ids, actions, log_probs, values, dones)])

        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        if self.wrap_with_accelerate:
            dataloader = self.accelerate.prepare(dataloader)

        # updating actor and critic

        self.actor.train()
        self.critic.train()

        for _ in tqdm(range(epochs), desc = 'learning actor/critic epoch', disable = self.quiet or not self.is_main_process):
            for (
                advantages,
                states,
                next_states,
                latent_gene_ids,
                actions,
                log_probs,
                old_values,
                dones
            ) in dataloader:

                if self.has_latent_genes:
                    latents = self.latent_gene_pool(latent_id = latent_gene_ids)

                    orig_latents = latents
                    latents = latents.detach()
                    latents.requires_grad_()
                else:
                    latents = None

                # learn actor

                distr = self.actor(states, latents)

                actor_loss = self.actor_loss(
                    distr, log_probs, actions, advantages,
                    use_spo = self.use_spo
                )

                # self-predictive representation - predict the next hidden from
                # the current hidden and action, against the ema target of the
                # next state - only over non-terminal transitions

                if self.use_hidden_spr:
                    spr = self.unwrap_model(self.hidden_spr)

                    hidden = self.unwrap_model(self.actor).latent(states, latents)
                    predicted = spr.predict(hidden, actions)
                    target = spr.target(next_states, latents)

                    transitions = ~dones.bool()

                    if transitions.any():
                        spr_loss = (2. - F.cosine_similarity(predicted[transitions], target[transitions], dim = -1)).mean()

                        actor_loss = actor_loss + self.hidden_spr_weight * spr_loss

                actor_loss.backward()

                if exists(self.has_grad_clip):
                    self.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                self.actor_optim.step()
                self.actor_optim.zero_grad()

                if self.use_hidden_spr:
                    self.hidden_spr_optim.step()
                    self.hidden_spr_optim.zero_grad()

                # learn critic with maybe classification loss

                critic_loss = self.unwrap_model(self.critic).forward_for_loss(
                    states,
                    latents,
                    old_values = old_values,
                    target = advantages + old_values,
                    clip_value = self.clip_value,
                    use_improved = self.use_improved_critic_loss,
                    **self.critic_loss_kwargs
                )

                critic_loss.backward()

                if exists(self.has_grad_clip):
                    self.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.critic_optim.step()
                self.critic_optim.zero_grad()

                # learn diversity discriminator

                diversity_discr_loss = self.zero

                if self.use_diversity_discr:
                    diversity_discr_logits = self.diversity_discr(states, next_states)
                    diversity_discr_loss = F.cross_entropy(diversity_discr_logits, latent_gene_ids)

                    diversity_discr_loss.backward()

                    if exists(self.has_grad_clip):
                        self.clip_grad_norm_(self.diversity_discr.parameters(), self.max_grad_norm)

                    self.diversity_discr_optim.step()
                    self.diversity_discr_optim.zero_grad()

                # log actor critic loss

                self.log(
                    actor_loss = actor_loss.item(),
                    critic_loss = critic_loss.item(),
                    diversity_discr_loss = diversity_discr_loss.item(),
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

            for _, states, *_ in tqdm(dataloader, desc = 'state norm learning', disable = self.quiet or not self.is_main_process):
                self.state_norm(states)

        # update warmed up state for discriminator

        if self.use_diversity_discr:
            self.has_diversity_discr_warmed_up.copy_(tensor(True))

        # update the hidden spr ema targets once per learn_from

        if self.use_hidden_spr:
            self.unwrap_model(self.hidden_spr).ema_update(self.unwrap_model(self.actor), self.hidden_spr_ema_update)

        # apply evolution

        should_update = False

        if self.has_latent_genes:
            should_update, _ = self.latent_gene_pool.genetic_algorithm_step(fitness_scores)

            if self.use_diversity_discr and should_update:
                self.has_diversity_discr_warmed_up.copy_(tensor(False))

                # reset discriminator

                unwrap = self.unwrap_model
                unwrap(self.diversity_discr).reset_parameters()

                # reset optimizer

                self.diversity_discr_optim.state.clear()

        # maybe shrink and perturb

        if self.should_noise_weights and divisible_by(self.step.item(), self.shrink_and_perturb_every):
            self.shrink_and_perturb_(self.actor)
            self.shrink_and_perturb_(self.critic)

        # increment step

        self.step.add_(1)

        return should_update, fitness_scores

# reinforcement learning related - ppo

def actor_loss(
    distr,
    old_log_probs,  # Float[b]
    actions,        # Int[b], or Float[b l] for beta
    advantages,     # Float[b]
    eps_clip = 0.2,
    entropy_weight = .01,
    eps = 1e-5,
    norm_advantages = True,
    use_spo = False
):
    batch = advantages.shape[0]

    log_probs = sum_to_batch(distr.log_prob(actions))
    entropy = sum_to_batch(distr.entropy())

    ratio = (log_probs - old_log_probs).exp()

    if norm_advantages:
        advantages = F.layer_norm(advantages, (batch,), eps = eps)

    if use_spo:
        # simple policy optimization - line 14 Algorithm 1 https://arxiv.org/abs/2401.16025v9

        actor_loss = - (
            ratio * advantages -
            advantages.abs() / (2 * eps_clip) * (ratio - 1.).square()
        )
    else:
        # classic clipped surrogate loss from ppo

        clipped_ratio = ratio.clamp(min = 1. - eps_clip, max = 1. + eps_clip)

        actor_loss = -torch.min(clipped_ratio * advantages, ratio * advantages)

    # add entropy loss for exploration

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
    use_state_norm = False,
    action_is_continuous = False, # continuous control - beta policy
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

    state_norm = StateNorm(dim = dim_state) if use_state_norm else None

    actor = Actor(
        num_actions = actor_num_actions,
        dim_state = dim_state,
        dim_latent = dim_latent,
        dim = actor_dim,
        mlp_depth = actor_mlp_depth,
        state_norm = state_norm,
        action_is_continuous = action_is_continuous,
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
    'next_state',
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

Slot = namedtuple('Slot', [
    'latent_id',
    'episode_id',
    'latent',
    'state',
    'time',
    'memories'
])

# rollout of episodes for each latent can be parallelized across the workers
# of an env implementing the vectorized interface (`num_envs`, `reset_one`,
# `step_batch`) - a gymnasium-style env (reset / step) is adapted to the
# vectorized interface with a single worker

def is_vectorized_env(env):
    return all(hasattr(env, attr) for attr in ('num_envs', 'reset_one', 'step_batch'))

class VectorizedEnvAdapter(Module):
    def __init__(
        self,
        env
    ):
        super().__init__()
        self.env = env
        self.num_envs = 1

    def reset_one(
        self,
        worker,
        seed = None
    ):
        assert worker == 0

        return self.env.reset(seed = seed)

    def step_batch(
        self,
        actions,
        worker_ids = None
    ):
        assert not exists(worker_ids) or list(worker_ids) == [0]

        action = np.asarray(actions[0])
        next_state, reward, terminated, truncated, *_ = self.env.step(action)

        return (
            np.asarray(next_state, dtype = np.float32)[None, ...],
            np.asarray(reward, dtype = np.float32)[None],
            np.asarray(terminated, dtype = bool)[None],
            np.asarray(truncated, dtype = bool)[None]
        )

def to_vectorized_env(env):
    return env if is_vectorized_env(env) else VectorizedEnvAdapter(env)

class EPO(Module):

    def __init__(
        self,
        agent: Agent,
        episodes_per_latent,
        max_episode_length,
        action_sample_temperature = 1.,
        fix_environ_across_latents = True,
        diversity_reward_weight = 1.,
    ):
        super().__init__()
        self.agent = agent
        self.action_sample_temperature = action_sample_temperature
        self.diversity_reward_weight = diversity_reward_weight

        self.num_latents = agent.latent_gene_pool.num_latents if agent.has_latent_genes else 1
        self.episodes_per_latent = episodes_per_latent
        self.max_episode_length = max_episode_length
        self.fix_environ_across_latents = fix_environ_across_latents

        self.register_buffer('dummy', tensor(0, device = agent.device))
        self.register_buffer('zero', tensor(0., device = agent.device))

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
        """rollout episodes for each latent - parallelized across the workers
        of a vectorized env, or one episode at a time on a single worker for a
        gymnasium-style env. when an episode ends, the next one from the
        rollout generator is loaded into that worker"""

        fix_environ_across_latents = default(fix_environ_across_latents, self.fix_environ_across_latents)

        env = to_vectorized_env(env)
        num_envs = env.num_envs

        self.agent.eval()

        invalid_episode = tensor(-1) # bootstrap transitions carry this id, to be discarded when learning
        num_episodes = self.num_latents * self.episodes_per_latent

        memories = memories if exists(memories) else []

        rewards_per_latent_episode = torch.zeros((self.num_latents, self.episodes_per_latent), device = self.device)

        rollout_gen = iter(self.rollouts_for_machine(fix_environ_across_latents))

        # slots - the episode each worker is currently rolling out

        slots: list[Slot | None] = [None] * num_envs

        def fill_slot(i):
            rollout = next(rollout_gen, None)
            if rollout is None:
                return

            latent_id, episode_id, maybe_seed = rollout

            latent = self.agent.unwrapped_latent_gene_pool(latent_id = latent_id) if self.agent.has_latent_genes else None

            seed = maybe_seed if fix_environ_across_latents else None
            state, _ = interface_torch_numpy(env.reset_one, device = self.device)(i, seed = seed)

            slots[i] = Slot(latent_id, episode_id, latent, state, 0, [])

        for i in range(num_envs):
            fill_slot(i)

        pbar = tqdm(total = num_episodes, desc = 'rollout', disable = self.agent.quiet or not self.agent.is_main_process)

        # each iteration, batch all active slots through the actor and critic,
        # then step the env - slots that finish are refilled and the batch
        # stays full until all episodes have been rolled out

        while num_episodes > 0:
            active = [i for i, slot in enumerate(slots) if exists(slot)]

            obs = stack([slots[i].state for i in active])
            latents = stack([slots[i].latent for i in active]) if self.agent.has_latent_genes else None

            actions, log_probs = self.agent.get_actor_actions(obs, latent = latents, sample = True, temperature = self.action_sample_temperature, use_unwrapped_model = True)
            values = self.agent.get_critic_values(obs, latent = latents, use_ema_if_available = True, use_unwrapped_model = True)

            next_obs, rewards, terminated, truncated = env.step_batch(actions.cpu().numpy(), worker_ids = active)[:4]

            rewards = from_numpy(np.asarray(rewards)).float().to(self.device)
            terminated = from_numpy(np.asarray(terminated)).to(self.device)
            truncated = from_numpy(np.asarray(truncated)).to(self.device)

            for k, worker in enumerate(active):
                slot = slots[worker]

                latent_id, episode_id, latent, state = slot.latent_id, slot.episode_id, slot.latent, slot.state

                next_state = from_numpy(np.array(next_obs[k])).float().to(self.device)

                reward, terminated_at, truncated_at = rewards[k], terminated[k], truncated[k]

                # maybe diversity reward for the latent

                if self.agent.use_diversity_discr and self.agent.has_diversity_discr_warmed_up.item():
                    with torch.no_grad():
                        logits = self.agent.unwrap_model(self.agent.diversity_discr)(rearrange(state, '... -> 1 ...'), rearrange(next_state, '... -> 1 ...'))

                        latent_log_probs = logits.log_softmax(dim = -1)
                        diversity_reward = latent_log_probs[0, latent_id] + math.log(self.agent.num_latents)

                        reward = reward + self.diversity_reward_weight * diversity_reward.item()

                done = truncated_at or terminated_at

                memory = Memory(
                    tensor(episode_id),
                    state,
                    next_state,
                    tensor(latent_id),
                    actions[k],
                    log_probs[k],
                    reward,
                    values[k],
                    terminated_at
                )

                memory = Memory(*tuple(t.cpu() for t in memory))

                slot.memories.append(memory)

                rewards_per_latent_episode[latent_id, episode_id] += reward

                # episode is done - bootstrap value if truncated, then refill

                if not done and slot.time + 1 < self.max_episode_length:
                    slots[worker] = slot._replace(state = next_state, time = slot.time + 1)
                    continue

                if not terminated_at:
                    next_value = temp_batch_dim(self.agent.get_critic_values)(next_state, latent = latent, use_ema_if_available = True, use_unwrapped_model = True)

                    memory = memory._replace(
                        episode_id = invalid_episode,
                        reward = next_value.cpu(),
                        value = next_value.cpu(),
                        done = tensor(True)
                    )

                    slot.memories.append(memory)

                memories.extend(slot.memories)

                num_episodes -= 1
                pbar.update(1)

                slots[worker] = None
                fill_slot(worker)

        pbar.close()

        return MemoriesAndCumulativeRewards(
            memories = memories,
            cumulative_rewards = rewards_per_latent_episode
        )

    def forward(
        self,
        env,
        num_learning_cycles,
        seed = None
    ):

        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

        for _ in tqdm(range(num_learning_cycles), desc = 'learning cycle', disable = self.agent.quiet or not self.agent.is_main_process):

            memories = self.gather_experience_from(env)

            self.agent.learn_from(memories)

        print('training complete')
