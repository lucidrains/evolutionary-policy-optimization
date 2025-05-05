import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.func import functional_call, vmap
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms as T

from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from evolutionary_policy_optimization.experimental import (
    crossover_weights,
    mutate_weight
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def divisible_by(num, den):
    return (num % den) == 0

#data

class MnistDataset(Dataset):
    def __init__(self, train):
        self.mnist = torchvision.datasets.MNIST('./data/mnist', train = train, download = True)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        return (digit_tensor / 255.).float().to(device), tensor(labels, device = device)

train_dataset = MnistDataset(train = True)
dl = DataLoader(train_dataset, batch_size = 32, shuffle = True)

eval_dataset = MnistDataset(train = False)
eval_dl = DataLoader(eval_dataset, batch_size = 32, shuffle = True)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# network

net = nn.Sequential(
    Rearrange('... c h w -> ... (c h w)'),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

# regular gradient descent

optim = Adam(net.parameters(), lr = 1e-3)

iter_train_dl = cycle(dl)
iter_eval_dl = cycle(eval_dl)

for i in range(1000):

    data, labels = next(iter_train_dl)

    logits = net(data)

    loss = F.cross_entropy(logits, labels)
    loss.backward()

    print(f'{i}: {loss.item():.3f}')

    optim.step()
    optim.zero_grad()

    if divisible_by(i + 1, 100):
        with torch.no_grad():
            eval_data, labels = next(iter_eval_dl)
            logits = net(eval_data)
            eval_loss = F.cross_entropy(logits, labels)

            total = labels.shape[0]
            correct = (logits.argmax(dim = -1) == labels).long().sum().item()

            print(f'{i}: eval loss: {eval_loss.item():.3f}')
            print(f'{i}: accuracy: {correct} / {total}')

# genetic algorithm on population of networks
# pop stands for population

pop_size = 100

params = dict(net.named_parameters())
pop_params = {name: (torch.randn((pop_size, *param.shape), device = device) * 1e-1) for name, param in params.items()}

def forward(params, data):
    return functional_call(net, params, data)

forward_pop_models = vmap(forward, in_dims = (0, None))

for i in range(1000):

    data, labels = next(iter_train_dl)

    pop_logits = forward_pop_models(pop_params, data)

    inv_fitnesses = F.cross_entropy(
        rearrange(pop_logits, 'p b l -> (p b) l'),
        repeat(labels, 'b -> (p b)', p = pop_size),
        reduction = 'none'
    )

    print(f'{i}: {inv_fitnesses.mean().item():.3f}')

# todo

# gradient descent and genetic algorithms

# todo
