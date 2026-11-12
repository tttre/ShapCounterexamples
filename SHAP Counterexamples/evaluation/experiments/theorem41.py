from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.SHAP.shapely_values import causal_shapely_values
from src.SHAP.interventional_contribution import causal_intervention_effect

### --------------------------------
### Sampler and Score Function Definitions
def causal_house_sampler_2(N, rng, fixed=None):
    fixed = {} if fixed is None else fixed

    # Sample the original distribution
    bedroom_noise = rng.poisson(2, size=N)
    area_noise = rng.normal(0, 10, size=N)

    # Assignment function for intervention on set "fixed"
    def assign(name, natural_values):
        if name in fixed:
            return np.full(N, fixed[name])
        return natural_values

    # Assign values to each feature, applying interventions if specified, recording causal effects
    bedrooms = assign("bedrooms",np.minimum(1 + bedroom_noise, 5),)

    area = assign("area",np.maximum(20 + 25 * bedrooms + area_noise, 20),)


    return {
        "bedrooms": bedrooms,
        "area": area,
    }

def house_price_score_2(features):
    return (
        2000 * features["area"]
    )
### --------------------------------

N = 100000
basis = [lambda features: features["area"], lambda features: features["bedrooms"]]

# Effect of changing bedrooms from 2 to 3 on each basis function.
d = np.array([
    causal_intervention_effect({"bedrooms": 2}, {"bedrooms": 3}, causal_house_sampler_2, psi, N)
    for psi in basis
])

# Feature order: bedrooms, area
X = [
    [2, 80]
]

for x in X:
    x = np.asarray(x, dtype=object)
    phi_causal = causal_shapely_values(x, causal_house_sampler_2, house_price_score_2, N)
    # Each column explains one basis function. Reverse the sampler's rows to (area, bedrooms).
    M = np.column_stack([
        causal_shapely_values(x, causal_house_sampler_2, psi, N)[::-1]
        for psi in basis
    ])
    print("----------------------------------------------------------------")
    print("SHAP Values for input:", x)
    print("M (rows and columns: area, bedrooms):")
    print(M)
    print("d (bedrooms 2 -> 3; basis: area, bedrooms):")
    print(d)
    M_augmented = np.vstack([M, d])  # Append d as a new row
    rank_M = np.linalg.matrix_rank(M)
    rank_augmented = np.linalg.matrix_rank(M_augmented)

    print(f"rank(M) = {rank_M}")
    print(f"rank([M; dᵀ]) = {rank_augmented}")
    print("----------------------------------------------------------------")
