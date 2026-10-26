### -------------------------------------
### A collection of distribution samplers
### -------------------------------------

import numpy as np
from .generators import binary_feature_generator, gaussian_feature_generator, uniform_feature_generator, count_feature_generator, categorical_feature_generator

def temp_sampler(N, rng):
    x1 = binary_feature_generator(p=0.5, N=N, seed=rng)
    x2 = gaussian_feature_generator(mean=0, std=1, N=N, seed=rng)
    return {
        "x1": x1,
        "x2": x2,
    }

def house_sampler(N, rng):
    bedrooms = np.minimum(
        1 + count_feature_generator(N=N, lam=2, seed=rng), 5
    )

    area = gaussian_feature_generator(
        mean=20 + 25 * bedrooms, std=10, N=N, seed=rng
    )
    area = np.maximum(area, 20)

    location = categorical_feature_generator(
        categories=["suburb", "city"], N=N, seed=rng
    )

    return {
        "bedrooms": bedrooms,
        "area": area,
        "location": location,
    }
