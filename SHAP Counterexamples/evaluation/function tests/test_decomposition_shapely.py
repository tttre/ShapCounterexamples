### ----------------------------------------------------------------
### Temporary testing file for exact Shapley decomposition implementation
### ----------------------------------------------------------------

import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.Data.Utils.score_functions import binary_score
from src.Data.Utils.generators import bernoulli_bit_generator
from src.SHAP.shapely_values import shapely_values
from src.SHAP.shapely_values_decomposition import shapely_values_decomposition


def feature_importance(X):
    decompositions = [
        shapely_values_decomposition(
            x, X, binary_score
        )
        for x in X
    ]

    A_values = np.array([A for A, B in decompositions])
    B_values = np.array([B for A, B in decompositions])

    phi = np.array([shapely_values(x, X, binary_score) for x in X])
    assert np.allclose(A_values + B_values, phi)

    I_A = np.abs(A_values).mean(axis=0)
    I_B = np.abs(B_values).mean(axis=0)
    I_total = np.abs(A_values + B_values).mean(axis=0)

    return I_total, I_A, I_B

# Generate reproducible inputs.
X = bernoulli_bit_generator(n_features=3, p=0.5, N=1000, seed=0)

I_total, I_A, I_B = feature_importance(X)

print("Total importance:", I_total)
print("A component magnitude:", I_A)
print("B component magnitude:", I_B)
