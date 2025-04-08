import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

# classes

class EPO(Module):
    def __init__(
        self
    ):
        super().__init__()
