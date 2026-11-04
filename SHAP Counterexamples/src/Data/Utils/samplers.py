### -------------------------------------
### A collection of distribution samplers
### -------------------------------------

import numpy as np
from .generators import binary_feature_generator, gaussian_feature_generator, uniform_feature_generator, count_feature_generator, categorical_feature_generator

### --------------------------------
### Conditional Samplers
def temp_sampler(N, rng):
    x1 = binary_feature_generator(p=0.5, N=N, seed=rng)
    x2 = gaussian_feature_generator(mean=0, std=1, N=N, seed=rng)
    return {
        "x1": x1,
        "x2": x2,
    }

def conditional_house_sampler(N, rng):
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
### --------------------------------

### --------------------------------
### Causal Samplers
def causal_house_sampler(N, rng, fixed=None):
    fixed = {} if fixed is None else fixed

    # Sample the original distribution
    bedroom_noise = rng.poisson(2, size=N)
    area_noise = rng.normal(0, 10, size=N)
    location_noise = rng.choice(["suburb", "city"], size=N)

    # Assignment function for intervention on set "fixed"
    def assign(name, natural_values):
        if name in fixed:
            return np.full(N, fixed[name])
        return natural_values

    # Assign values to each feature, applying interventions if specified, recording causal effects
    bedrooms = assign("bedrooms",np.minimum(1 + bedroom_noise, 5),)

    area = assign("area",np.maximum(20 + 25 * bedrooms + area_noise, 20),)

    location = assign("location", location_noise)

    return {
        "bedrooms": bedrooms,
        "area": area,
        "location": location,
    }
### --------------------------------