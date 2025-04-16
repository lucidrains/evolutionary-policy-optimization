<img width="450px" alt="fig1" src="https://github.com/user-attachments/assets/33bef569-e786-4f09-bdee-56bad7ea9e6d" />

## Evolutionary Policy Optimization (wip)

Pytorch implementation of [Evolutionary Policy Optimization](https://web3.arxiv.org/abs/2503.19037), from Wang et al. of the Robotics Institute at Carnegie Mellon University

This paper stands out, as I have witnessed the positive effects first hand in an [exploratory project](https://github.com/lucidrains/firefly-torch) (mixing evolution with gradient based methods). Perhaps the Alexnet moment for genetic algorithms has not come to pass yet.

Besides their latent variable strategy, I'll also throw in some attempts with crossover in weight space

Update: I see, mixing genetic algorithms with gradient based method is already a research field, under [Memetic algorithms](https://en.wikipedia.org/wiki/Memetic_algorithm)

## Install

```bash
$ pip install evolutionary-policy-optimization
```

## Usage

```python
import torch

from evolutionary_policy_optimization import (
    LatentGenePool,
    Actor,
    Critic
)

latent_pool = LatentGenePool(
    num_latents = 128,
    dim_latent = 32,
)

state = torch.randn(1, 512)

actor = Actor(512, dim_hiddens = (256, 128), num_actions = 4, dim_latent = 32)
critic = Critic(512, dim_hiddens = (256, 128, 64), dim_latent = 32)

latent = latent_pool(latent_id = 2)

actions = actor(state, latent)
value = critic(state, latent)

# interact with environment and receive rewards, termination etc

# derive a fitness score for each gene / latent

fitness = torch.randn(128)

latent_pool.genetic_algorithm_step(fitness) # update latent genes with genetic algorithm
```

## Citations

```bibtex
@inproceedings{Wang2025EvolutionaryPO,
    title = {Evolutionary Policy Optimization},
    author = {Jianren Wang and Yifan Su and Abhinav Gupta and Deepak Pathak},
    year  = {2025},
    url   = {https://api.semanticscholar.org/CorpusID:277313729}
}
```

```bibtex
@article{Farebrother2024StopRT,
    title   = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
    author  = {Jesse Farebrother and Jordi Orbay and Quan Ho Vuong and Adrien Ali Taiga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
    journal = {ArXiv},
    year   = {2024},
    volume = {abs/2403.03950},
    url    = {https://api.semanticscholar.org/CorpusID:268253088}
}
```

*Evolution is cleverer than you are.* - Leslie Orgel
