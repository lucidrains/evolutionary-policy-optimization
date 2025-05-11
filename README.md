<img width="450px" alt="fig1" src="https://github.com/user-attachments/assets/33bef569-e786-4f09-bdee-56bad7ea9e6d" />

## Evolutionary Policy Optimization

Pytorch implementation of [Evolutionary Policy Optimization](https://web3.arxiv.org/abs/2503.19037), from [Wang](https://www.jianrenw.com/) et al. of the Robotics Institute at Carnegie Mellon University

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
    num_latents = 16,
    dim_latent = 32,
    actor_num_actions = 5,
    actor_dim = 256,
    actor_mlp_depth = 2,
    critic_dim = 256,
    critic_mlp_depth = 3,
    latent_gene_pool_kwargs = dict(
        frac_natural_selected = 0.5
    )
)

epo = EPO(
    agent,
    episodes_per_latent = 1,
    max_episode_length = 10,
    action_sample_temperature = 1.
)

env = Env((512,))

epo(agent, env, num_learning_cycles = 5)

# saving and loading

agent.save('./agent.pt', overwrite = True)
agent.load('./agent.pt')
```

## Contributing

At the project root, run

```bash
$ pip install '.[test]' # or `uv pip install '.[test]'`
```

Then add your tests to `tests/test_epo.py` and run

```bash
$ pytest tests/
```

That's it

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

```bibtex
@article{Banerjee2022BoostingEI,
    title   = {Boosting Exploration in Actor-Critic Algorithms by Incentivizing Plausible Novel States},
    author  = {Chayan Banerjee and Zhiyong Chen and Nasimul Noman},
    journal = {2023 62nd IEEE Conference on Decision and Control (CDC)},
    year    = {2022},
    pages   = {7009-7014},
    url     = {https://api.semanticscholar.org/CorpusID:252682944}
}
```

```bibtex
@article{Doerr2017FastGA,
    title   = {Fast genetic algorithms},
    author  = {Benjamin Doerr and Huu Phuoc Le and R{\'e}gis Makhmara and Ta Duy Nguyen},
    journal = {Proceedings of the Genetic and Evolutionary Computation Conference},
    year    = {2017},
    url     = {https://api.semanticscholar.org/CorpusID:16196841}
}
```

```bibtex
@article{Lee2024AnalysisClippedCritic
    title   = {On Analysis of Clipped Critic Loss in Proximal Policy Gradient},
    author  = {Yongjin Lee, Moonyoung Chung},
    journal = {Authorea},
    year    = {2024}
}
```

```bibtex
@article{Ash2019OnTD,
    title   = {On the Difficulty of Warm-Starting Neural Network Training},
    author  = {Jordan T. Ash and Ryan P. Adams},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1910.08475},
    url     = {https://api.semanticscholar.org/CorpusID:204788802}
}
```

```bibtex
@inproceedings{Gerasimov2025YouDN,
    title   = {You Do Not Fully Utilize Transformer's Representation Capacity},
    author  = {Gleb Gerasimov and Yaroslav Aksenov and Nikita Balagansky and Viacheslav Sinii and Daniil Gavrilov},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276317819}
}
```

```bibtex
@article{Lee2024SimBaSB,
    title   = {SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning},
    author  = {Hojoon Lee and Dongyoon Hwang and Donghu Kim and Hyunseung Kim and Jun Jet Tai and Kaushik Subramanian and Peter R. Wurman and Jaegul Choo and Peter Stone and Takuma Seno},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.09754},
    url     = {https://api.semanticscholar.org/CorpusID:273346233}
}
```

```bibtex
@article{Karras2019stylegan2,
    title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
    author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
    journal = {CoRR},
    volume  = {abs/1912.04958},
    year    = {2019},
}
```

*Evolution is cleverer than you are.* - Leslie Orgel
