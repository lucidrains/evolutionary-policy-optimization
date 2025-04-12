import torch

def crossover_weights(w1, w2, transpose = False):
    assert w2.shape == w2.shape
    assert w1.ndim == 2

    if transpose:
        w1, w2 = w1.t(), w2.t()

    rank = min(w2.shape)
    assert rank >= 2

    u1, s1, v1 = torch.svd(w1)
    u2, s2, v2 = torch.svd(w2)

    mask = torch.randperm(rank) < (rank // 2)

    u = torch.where(mask[None, :], u1, u2)
    s = torch.where(mask, s1, s2)
    v = torch.where(mask[None, :], v1, v2)

    out = u @ torch.diag_embed(s) @ v.mT

    if transpose:
        out = out.t()

    return out
