<img width="450px" alt="fig1" src="https://github.com/user-attachments/assets/33bef569-e786-4f09-bdee-56bad7ea9e6d" />

## Evolutionary Policy Optimization (wip)

Pytorch implementation of [Evolutionary Policy Optimization](https://web3.arxiv.org/abs/2503.19037), from Wang et al. of the Robotics Institute at Carnegie Mellon University

This paper stands out, as I have witnessed the positive effects first hand in an [exploratory project](https://github.com/lucidrains/firefly-torch) (mixing evolution with gradient based methods). Perhaps the Alexnet moment for genetic algorithms has not come to pass yet.

Besides their latent variable strategy, I'll also throw in some attempts with crossover in weight space

Update: I see, mixing genetic algorithms with gradient based method is already a research field, under [Memetic algorithms](https://en.wikipedia.org/wiki/Memetic_algorithm). This is also incidentally what I have concluded what Science is. I am in direct exposure to this phenomenon on a daily basis

## Usage

```python
import torch

from evolutionary_policy_optimization import (
    LatentGenePool,
    MLP
)

latent_pool = LatentGenePool(
    num_latents = 32,
    dim_latent = 32,
    net = MLP(
        dims = (512, 256),
        dim_latent = 32,
    )
)

state = torch.randn(1, 512)
action = latent_pool(state, latent_id = 3) # use latent / gene 4

# interact with environment and receive rewards, termination etc

# derive a fitness score for each gene / latent

fitness = torch.randn(32)

latent_pool.genetic_algorithm_step(fitness) # update latents using one generation of genetic algorithm

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

*Evolution is cleverer than you are.* - Leslie Orgel
