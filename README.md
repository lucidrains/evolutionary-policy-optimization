<img width="450px" alt="fig1" src="https://github.com/user-attachments/assets/33bef569-e786-4f09-bdee-56bad7ea9e6d" />

## Evolutionary Policy Optimization (wip)

Pytorch implementation of [Evolutionary Policy Optimization](https://web3.arxiv.org/abs/2503.19037), from Wang et al. of the Robotics Institute at Carnegie Mellon University

This paper stands out, as I have witnessed the positive effects first hand in an [exploratory project](https://github.com/lucidrains/firefly-torch) (mixing evolution with gradient based methods). Perhaps the Alexnet moment for genetic algorithms has not come to pass yet.

Besides their latent variable strategy, I'll also throw in some attempts with crossover in weight space

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

End to end learning

```python
import torch

from evolutionary_policy_optimization import (
    create_agent,
    EPO,
    Env
)

agent = create_agent(
    dim_state = 512,
    num_latents = 8,
    dim_latent = 32,
    actor_num_actions = 5,
    actor_dim_hiddens = (256, 128),
    critic_dim_hiddens = (256, 128, 64)
)

epo = EPO(
    agent,
    episodes_per_latent = 1,
    max_episode_length = 10,
    action_sample_temperature = 1.
)

env = Env((512,))

memories = epo(env)

agent(memories)

# saving and loading

agent.save('./agent.pt', overwrite = True)
agent.load('./agent.pt')
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

```bibtex
@inproceedings{Khadka2018EvolutionGuidedPG,
    title   = {Evolution-Guided Policy Gradient in Reinforcement Learning},
    author  = {Shauharda Khadka and Kagan Tumer},
    booktitle = {Neural Information Processing Systems},
    year    = {2018},
    url     = {https://api.semanticscholar.org/CorpusID:53096951}
}
```

```bibtex
@article{Fortunato2017NoisyNF,
    title   = {Noisy Networks for Exploration},
    author  = {Meire Fortunato and Mohammad Gheshlaghi Azar and Bilal Piot and Jacob Menick and Ian Osband and Alex Graves and Vlad Mnih and R{\'e}mi Munos and Demis Hassabis and Olivier Pietquin and Charles Blundell and Shane Legg},
    journal = {ArXiv},
    year    = {2017},
    volume  = {abs/1706.10295},
    url     = {https://api.semanticscholar.org/CorpusID:5176587}
}
```

*Evolution is cleverer than you are.* - Leslie Orgel
