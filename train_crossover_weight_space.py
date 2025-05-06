from random import uniform

import torch
from torch import nn, tensor, randn
import torch.nn.functional as F
from torch.func import functional_call, vmap
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms as T

from einops.layers.torch import Rearrange
from einops import repeat, rearrange, reduce, pack, unpack

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

batch = 32

train_dataset = MnistDataset(train = True)
dl = DataLoader(train_dataset, batch_size = batch, shuffle = True, drop_last = True)

eval_dataset = MnistDataset(train = False)
eval_dl = DataLoader(eval_dataset, batch_size = batch, shuffle = True, drop_last = True)

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# network

net = nn.Sequential(
    Rearrange('... c h w -> ... (c h w)'),
    nn.Linear(784, 64, bias = False),
    nn.ReLU(),
    nn.Linear(64, 10, bias = False),
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

# periodic crossover from genetic algorithm on population of networks
# pop stands for population

pop_size = 100
num_selected = 25
num_offsprings = pop_size - num_selected
tournament_size = 5
learning_rate = 1e-3

params = dict(net.named_parameters())
pop_params = {name: (randn((pop_size, *param.shape), device = device) * 0.1).requires_grad_() for name, param in params.items()}

optim = Adam(pop_params.values(), lr = learning_rate)

def forward(params, data):
    return functional_call(net, params, data)

forward_pop_models = vmap(forward, in_dims = (0, None))

for i in range(1000):

    data, labels = next(iter_train_dl)

    pop_logits = forward_pop_models(pop_params, data)

    losses = F.cross_entropy(
        rearrange(pop_logits, 'p b l -> b l p'),
        repeat(labels, 'b -> b p', p = pop_size),
        reduction = 'none'
    )

    losses.sum(dim = -1).mean(dim = 0).backward()

    print(f'{i}: loss: {losses.mean().item():.3f}')

    optim.step()
    optim.zero_grad()

    # evaluate

    if divisible_by(i + 1, 100):

        with torch.no_grad():

            eval_data, labels = next(iter_eval_dl)
            pop_logits = forward_pop_models(pop_params, eval_data)

            eval_loss = F.cross_entropy(
                rearrange(pop_logits, 'p b l -> b l p'),
                repeat(labels, 'b -> b p', p = pop_size),
                reduction = 'none'
            )

            total = labels.shape[0] * pop_size
            correct = (pop_logits.argmax(dim = -1) == labels).long().sum().item()

            print(f'{i}: eval loss: {eval_loss.mean().item():.3f}')
            print(f'{i}: accuracy: {correct} / {total}')

            # genetic algorithm on population

            fitnesses = 1. / eval_loss
            
            fitnesses = reduce(fitnesses, 'b p -> p', 'mean') # average across samples

            # selection

            sel_fitnesses, sel_indices = fitnesses.topk(num_selected, dim = -1)

            sel_parents = {name: (param[sel_indices]) for name, param in pop_params.items()}

            # tournaments

            tourn_ids = randn((num_offsprings, tournament_size)).argsort(dim = -1)
            tourn_scores = sel_fitnesses[tourn_ids]

            parent_ids = tourn_scores.topk(2, dim = -1).indices
            parent_ids = rearrange(parent_ids, 'offsprings couple -> couple offsprings')

            # crossover

            for param, sel_parent_param in zip(pop_params.values(), sel_parents.values()):
                parent1, parent2 = sel_parent_param[parent_ids]

                children = parent1.lerp_(parent2, uniform(0.25, 0.75))

                pop = torch.cat((sel_parent_param, children))

                param.data.copy_(pop).requires_grad_()

            # new optim

            optim = Adam(pop_params.values(), lr = learning_rate)
