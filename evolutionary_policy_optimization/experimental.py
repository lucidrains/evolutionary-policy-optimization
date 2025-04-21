import torch
from einops import rearrange

def crossover_weights(w1, w2, transpose = False):
    assert w2.shape == w2.shape

    no_batch = w1.ndim == 2

    if no_batch:
        w1, w2 = tuple(rearrange(t, '... -> 1 ...') for t in (w1, w2))

    assert w1.ndim == 3

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
    v = torch.where(mask[:, None, :], v1, v2)

    out = u @ torch.diag_embed(s) @ v.mT

    if transpose:
        out = rearrange(out, 'b j i -> b i j')

    if no_batch:
        out = rearrange(out, '1 ... -> ...')

    return out

def mutate_weight(
    w,
    transpose = False,
    mutation_strength = 1.
):

    if transpose:
        w = w.transpose(-1, -2)

    rank = min(w2.shape[1:])
    assert rank >= 2

    u, s, v = torch.svd(w)
    u = u + torch.randn_like(u) * mutation_strength
    v = v + torch.randn_like(v) * mutation_strength

    out = u @ torch.diag_embed(s) @ v.mT

    if transpose:
        out = out.transpose(-1, -2)

    return out

if __name__ == '__main__':
    w1 = torch.randn(32, 16)
    w2 = torch.randn(32, 16)

    child = crossover_weights(w1, w2)
    mutated_w1 = mutate_weight(w1)

    assert child.shape == w2.shape
