from random import uniform
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.func import vmap, functional_call
from torch.nn import Module, ParameterList

from einops import rearrange, reduce, repeat

def exists(v):
    return v is not None

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def crossover_weights(w1, w2):
    assert w2.shape == w2.shape

    no_batch = w1.ndim == 2

    if no_batch:
        w1, w2 = tuple(rearrange(t, '... -> 1 ...') for t in (w1, w2))

    assert w1.ndim == 3

    i, j = w1.shape[-2:]
    transpose = i < j

    if transpose:
        w1, w2 = tuple(rearrange(t, 'b i j -> b j i') for t in (w1, w2))

    rank = min(w2.shape[1:])
    assert rank >= 2

    batch = w1.shape[0]

    u1, s1, v1 = torch.svd(w1)
    u2, s2, v2 = torch.svd(w2)

    batch_randperm = torch.randn((batch, rank), device = w1.device).argsort(dim = -1)
    mask = batch_randperm < (rank // 2)

    u = torch.where(mask[:, None, :], u1, u2)
    s = torch.where(mask, s1, s2)
    v = torch.where(mask[:, :, None], v1, v2)

    out = u @ torch.diag_embed(s) @ v.mT

    if transpose:
        out = rearrange(out, 'b j i -> b i j')

    if no_batch:
        out = rearrange(out, '1 ... -> ...')

    return out

def mutate_weight(
    w,
    mutation_strength = 1.
):

    i, j = w.shape[-2:]
    transpose = i < j

    if transpose:
        w = w.transpose(-1, -2)

    rank = min(w.shape[1:])
    assert rank >= 2

    u, s, v = torch.svd(w)

    u = u + torch.randn_like(u) * mutation_strength
    v = v + torch.randn_like(v) * mutation_strength

    u = l2norm(u, dim = -2)
    v = l2norm(v, dim = -1)

    out = u @ torch.diag_embed(s) @ v.mT

    if transpose:
        out = out.transpose(-1, -2)

    return out

# wrapper that manages network to population
# able to receive fitness and employ selection + crossover

class PopulationWrapper(Module):
    def __init__(
        self,
        net: Module,
        pop_size,
        num_selected,
        tournament_size,
        learning_rate = 1e-3,
        init_std_dev = 1e-1
    ):
        super().__init__()
        assert num_selected < pop_size
        assert tournament_size < num_selected

        self.pop_size = pop_size
        self.num_selected = num_selected
        self.tournament_size = tournament_size
        self.num_offsprings = pop_size - num_selected

        self.net = net

        params = dict(net.named_parameters())
        device = next(iter(params.values())).device

        pop_params = {name: (torch.randn((pop_size, *param.shape), device = device) * init_std_dev).requires_grad_() for name, param in params.items()}

        self.param_names = pop_params.keys()
        self.param_values = ParameterList(list(pop_params.values()))

        def _forward(params, data):
            return functional_call(net, params, data)

        self.forward_pop_nets = vmap(_forward, in_dims = (0, None))

    @property
    def pop_params(self):
        return dict(zip(self.param_names, self.param_values))

    def individual(self, id) -> Module:
        assert 0 <= id < self.pop_size
        state_dict = {key: param[id] for key, param in self.pop_params.items()}

        net = deepcopy(self.net)
        net.load_state_dict(state_dict)
        return net

    def parameters(self):
        return self.pop_params.values()

    def genetic_algorithm_step_(
        self,
        fitnesses
    ):
        fitnesses = reduce(fitnesses, 'b p -> p', 'mean') # average across samples

        num_selected = self.num_selected

        # selection

        sel_fitnesses, sel_indices = fitnesses.topk(num_selected, dim = -1)

        # tournaments

        tourn_ids = torch.randn((self.num_offsprings, self.tournament_size)).argsort(dim = -1)
        tourn_scores = sel_fitnesses[tourn_ids]

        winner_ids = tourn_scores.topk(2, dim = -1).indices
        winner_ids = rearrange(winner_ids, 'offsprings couple -> couple offsprings')
        parent_ids = sel_indices[winner_ids]

        # crossover

        for param in self.param_values:
            parents = param[sel_indices]
            parent1, parent2 = param[parent_ids]

            children = parent1.lerp_(parent2, uniform(0.25, 0.75))

            pop = torch.cat((parents, children))

            param.data.copy_(pop)

    def forward(
        self,
        data,
        *,
        individual_id = None,
        labels = None,
        return_logits_with_loss = False
    ):
        # if `individual_id` passed in, will forward for only that one network

        if exists(individual_id):
            assert 0 <= individual_id < self.pop_size
            params = {key: param[individual_id] for key, param in self.pop_params.items()}
            return functional_call(self.net, params, data)

        out = self.forward_pop_nets(dict(self.pop_params), data)

        if not exists(labels):
            return out

        logits = out
        pop_size = logits.shape[0]

        losses = F.cross_entropy(
            rearrange(logits, 'p b ... l -> (p b) l ...'),
            repeat(labels, 'b ... -> (p b) ...', p = pop_size),
            reduction = 'none'
        )

        losses = rearrange(losses, '(p b) ... -> p b ...', p = pop_size)

        if not return_logits_with_loss:
            return losses

        return losses, logits

# test

if __name__ == '__main__':
    w1 = torch.randn(2, 32, 16)
    w2 = torch.randn(2, 32, 16)

    child = crossover_weights(w1, w2)
    mutated_w1 = mutate_weight(w1)

    assert child.shape == w2.shape
