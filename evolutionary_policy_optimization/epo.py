from __future__ import annotations

import torch
from torch import nn, cat

import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def xnor(x, y):
    return not (x ^ y)

# crossover

def crossover_weights(w1, w2):
    assert w2.shape == w2.shape
    assert w1.ndim == 2

    rank = min(w2.shape)
    assert rank >= 2

    u1, s1, v1 = torch.svd(w1)
    u2, s2, v2 = torch.svd(w2)

    mask = torch.randperm(rank) < (rank // 2)

    u = torch.where(mask[None, :], u1, u2)
    s = torch.where(mask, s1, s2)
    v = torch.where(mask[None, :], v1, v2)

    return u @ torch.diag_embed(s) @ v.mT

def crossover_latents(
    parent1, parent2,
    weight = None,
    random = False
):
    assert parent1.shape == parent2.shape

    if random:
        assert not exists(weight)
        weight = torch.randn_like(parent1).sigmoid()
    else:
        weight = default(weight, 0.5) # they do a simple averaging for the latents as crossover, but allow for random interpolation, as well extend this work for tournament selection, where same set of parents may be re-selected

    child = torch.lerp(parent1, parent2, weight)
    return child

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
        first_dim += dim_latent
        dims = (first_dim, *rest_dims)

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
        assert xnor(self.needs_latent, exists(latent))

        if exists(latent):
            # start with naive concatenative conditioning
            # but will also offer some alternatives once a spark is seen (film, adaptive linear from stylegan, etc)

            batch = x.shape[0]

            latent = self.encode_latent(latent)
            latent = repeat(latent, 'd -> b d', b = batch)

            x = cat((x, latent), dim = -1)

        # layers

        for ind, layer in enumerate(self.layers, start = 1):
            is_last = ind == len(self.layers)

            x = layer(x)

            if not is_last:
                x = F.silu(x)

        return x

# classes

class EPO(Module):
    def __init__(
        self
    ):
        super().__init__()
