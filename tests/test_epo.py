import torch

from evolutionary_policy_optimization import (
    LatentGenePool,
    MLP
)

def test_readme():

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
