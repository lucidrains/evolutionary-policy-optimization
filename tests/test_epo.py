import pytest

import torch
from evolutionary_policy_optimization import LatentGenePool

@pytest.mark.parametrize('num_latent_sets', (1, 4))
def test_readme(
    num_latent_sets
):

    latent_pool = LatentGenePool(
        num_latents = 32,
        dim_latent = 32,
        num_latent_sets = 4,
        net = dict(
            dims = (512, 256)
        )
    )

    state = torch.randn(1, 512)
    action = latent_pool(state, latent_id = 3) # use latent / gene 4

    # interact with environment and receive rewards, termination etc

    # derive a fitness score for each gene / latent

    fitness = torch.randn(32)

    latent_pool.genetic_algorithm_step(fitness) # update latents using one generation of genetic algorithm
