from random import uniform

import torch
from torch import nn, tensor, randn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms as T

from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from evolutionary_policy_optimization.experimental import PopulationWrapper

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
learning_rate = 3e-4

pop_net = PopulationWrapper(
    net,
    pop_size = pop_size,
    num_selected = 25,
    tournament_size = 5,
    learning_rate = 1e-3
)

optim = Adam(pop_net.parameters(), lr = learning_rate)

for i in range(1000):
    pop_net.train()

    data, labels = next(iter_train_dl)

    losses = pop_net(data, labels = labels)

    losses.sum(dim = 0).mean().backward()

    print(f'{i}: loss: {losses.mean().item():.3f}')

    optim.step()
    optim.zero_grad()

    # evaluate

    if divisible_by(i + 1, 100):

        with torch.no_grad():

            pop_net.eval()

            eval_data, labels = next(iter_eval_dl)
            eval_loss, logits = pop_net(eval_data, labels = labels, return_logits_with_loss = True)

            total = labels.shape[0] * pop_size
            correct = (logits.argmax(dim = -1) == labels).long().sum().item()

            print(f'{i}: eval loss: {eval_loss.mean().item():.3f}')
            print(f'{i}: accuracy: {correct} / {total}')

            # genetic algorithm on population

            fitnesses = 1. / eval_loss

            pop_net.genetic_algorithm_step_(fitnesses)
            
            # new optim

            optim = Adam(pop_net.parameters(), lr = learning_rate)
