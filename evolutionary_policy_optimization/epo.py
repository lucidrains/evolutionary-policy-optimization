import torch
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class EPO(Module):
    def __init__(
        self
    ):
        super().__init__()
