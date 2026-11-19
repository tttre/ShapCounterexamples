from pathlib import Path
import sys
import numpy as np
from scipy.linalg import orth
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.SHAP.Utils.value_functions import causal_value_function
from src.SHAP.shapely_values import causal_shapely_values
from src.SHAP.interventional_contribution import causal_intervention_effect
from src.SCM.Utils.graphs import SCM
from src.SCM.Utils.nodes import Node
from src.SCM.SCM_coefficient_matrix import scm_coefficient_matrix

### Sampler
def example_sampler(N, rng, fixed=None):
    fixed = {} if fixed is None else fixed

    # Sample the original distribution
    noise1 = rng.binomial(1,p=0.5, size=N)
    noise2 = rng.binomial(1,p=0.25, size=N)

    # Assignment function for intervention on set "fixed"
    def assign(name, natural_values):
        if name in fixed:
            return np.full(N, fixed[name])
        return natural_values

    # Assign values to each feature, applying interventions if specified, recording causal effects
    X1 = assign("X1",noise1)
    X2 = assign("X2",np.logical_xor(noise2,X1))


    return {
        "X1": X1,
        "X2": X2,
    }

### Basis Functions: (1,X1,X_2,X_1*X_2), Input: (x1,x2) = (1,1)
### Compute V matrix for the basis functions and inputs
basis = [
    lambda features: np.ones_like(features["X1"]),
    lambda features: features["X1"],
    lambda features: features["X2"],
    lambda features: features["X1"] * features["X2"]
]

coalitions = [
    {},
    {"X1": 1},
    {"X2": 1},
    {"X1": 1, "X2": 1}
]

V = np.array([
    [causal_value_function(coalition, example_sampler, psi, N=100000, seed=0) for psi in basis] for coalition in coalitions
])

if np.linalg.matrix_rank(V) == V.shape[0]:
    print("V is full rank")
else:
    print("V is not full rank")

### Graph Function
graph = SCM([
    Node("X1"),
    Node("X2", parents=("X1",)),
    Node("f", parents=("X1", "X2")),
])

W, labels = scm_coefficient_matrix(
    graph,
    features=("X1", "X2"),
    target="f",
)


### Compute the colspan of W.T, meaning every perfect intervention whose intervention effect 
### can be reconstructed from the causal SHAP attribution vector

Q = orth(W.T)
print("col(W.T): ", Q)

### Prediction Function
def f(features):
    return 1 + 2 * features["X1"] + 3 * features["X2"] + 4 * features["X1"] * features["X2"]

### Cross-Feature Intervention: I1 = do(X1=1) -> I2 = do(X2=1)
r = np.array([0, -1, 1, 0])  # I2 - I1 = E_(do(X2=1))(f) - E_do(X1=1)(f) = E_(do(X2=1))(f) - E(f) - E_do(X1=1)(f) + E(f) = Perfect Intervention Effect of I2 - Perfect Intervention Effect of I1
lambdas = np.linalg.lstsq(W.T, r, rcond=None)[0]
assert np.allclose(W.T @ lambdas, r) # Solves W.T @ lambdas = r, if it can be reconstructed

phi = causal_shapely_values(np.array([1, 1]), example_sampler, f, N=100000, seed=0)
effect = causal_intervention_effect({"X1": 1}, {"X2": 1}, example_sampler, f, N=100000, seed=0)
reconstructed_effect = lambdas @ phi
assert np.allclose(effect, reconstructed_effect)

print("Causal intervention effect:", effect)
print("Reconstructed causal intervention effect:", reconstructed_effect)
