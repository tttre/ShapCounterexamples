from pathlib import Path
import sys
import numpy as np
from scipy.linalg import orth
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.SHAP.Utils.value_functions import causal_value_function
from src.SHAP.shapely_values import causal_shapely_values
from src.SHAP.interventional_contribution import causal_intervention_effect
from src.Data.Utils.generators import uniform_bit_generator
from src.SCM.Utils.graphs import SCM
from src.SCM.Utils.nodes import Node
from src.SCM.SCM_coefficient_matrix import scm_coefficient_matrix

### Sampler
def example_sampler(N, rng, fixed=None):
    fixed = {} if fixed is None else fixed

    # Sample the original distribution
    noise1 = rng.binomial(1,p=0.5, size=N)
    noise2 = rng.binomial(1,p=0.5, size=N)
    noise3 = rng.binomial(1,p=0.5, size=N)

    # Assignment function for intervention on set "fixed"
    def assign(name, natural_values):
        if name in fixed:
            return np.full(N, fixed[name])
        return natural_values

    # Assign values to each feature, applying interventions if specified, recording causal effects
    X1 = assign("X1",noise1)
    X2 = assign("X2",noise2)
    X3 = assign("X3",noise3)


    return {
        "X1": X1,
        "X2": X2,
        "X3": X3,
    }

### Basis Functions: (1,X1,X2,X3,X1*X2,X1*X3,X2*X3,X1*X2*X3), Input: (x1,x2,x3) = (1,1,1)
### Compute V matrix for the basis functions and inputs
basis = [
    lambda features: np.ones_like(features["X1"]),
    lambda features: features["X1"],
    lambda features: features["X2"],
    lambda features: features["X3"],
    lambda features: features["X1"] * features["X2"],
    lambda features: features["X1"] * features["X3"],
    lambda features: features["X2"] * features["X3"],
    lambda features: features["X1"] * features["X2"] * features["X3"]
]

coalitions = [
    {},
    {"X1": 1},
    {"X2": 1},
    {"X3": 1},
    {"X1": 1, "X2": 1},
    {"X1": 1, "X3": 1},
    {"X2": 1, "X3": 1},
    {"X1": 1, "X2": 1, "X3": 1}
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
    Node("X2"),
    Node("X3"),
    Node("f", parents=("X1", "X2", "X3")),
])

W, labels = scm_coefficient_matrix(
    graph,
    features=("X1", "X2", "X3"),
    target="f",
)


### Compute the colspan of W.T, meaning every perfect intervention whose intervention effect 
### can be reconstructed from the causal SHAP attribution vector

Q = orth(W.T)

### Prediction Function
# Exact expectations give SHAP = (0,0,0), but CIE(I1 -> I2) = 1.
def f(features):
    return (features["X1"] - features["X2"]) * (4 * features["X3"] - 3)

### Counterexample
### Cross-Feature Intervention: I1 = do(X1=1) -> I2 = do(X2=1)
print("Counterexample ------------------------------------------------------")
r = np.array([0, -1, 1, 0, 0, 0, 0, 0])  # I2 - I1 = E_(do(X2=1))(f) - E_do(X1=1)(f) = E_(do(X2=1))(f) - E(f) - E_do(X1=1)(f) + E(f) = Perfect Intervention Effect of I2 - Perfect Intervention Effect of I1

projection = Q @ (Q.T @ r)
outside = not np.allclose(r, projection, atol=1e-10, rtol=0)

print("r is outside col(W.T):", outside)

phi = causal_shapely_values(np.array([1, 1, 1]), example_sampler, f, N=100000, seed=0)
effect = causal_intervention_effect({"X1": 1},{"X2": 1}, example_sampler, f, N=100000, seed=0)

print("Shapley values:", phi)
print("Causal intervention effect:", effect)
print("---------------------------------------------------------------------")
### Working Example
### Cross-Feature Intervention: I1 = do({}) -> I2 = do(X1=1,X2=1,X3=1)
print("Working Example -----------------------------------------------------")
r = np.array([-1, 0, 0, 0, 0, 0, 0, 1])

projection = Q @ (Q.T @ r)
outside = not np.allclose(r, projection, atol=1e-10, rtol=0)

print("r is outside col(W.T):", outside)
print("Reconstruction is possible")

lambdas = np.linalg.lstsq(W.T, r, rcond=None)[0]

phi = causal_shapely_values(np.array([1, 1, 1]), example_sampler, f, N=100000, seed=0)
effect = causal_intervention_effect({},{"X1": 1,"X2": 1,"X3": 1}, example_sampler, f, N=100000, seed=0)
reconstructed_effect = lambdas @ phi

print("Causal intervention effect:", effect)
print("Reconstructed causal intervention effect:", reconstructed_effect)
print("---------------------------------------------------------------------")

### Exact Check: Enumerate the eight equally likely states of the independent binary roots
def exact_sampler(N, rng, fixed=None):
    fixed = {} if fixed is None else fixed
    points = uniform_bit_generator(3)  # N and rng are unused for exact enumeration.
    return {
        name: np.full(len(points), fixed[name]) if name in fixed else points[:, i]
        for i, name in enumerate(("X1", "X2", "X3"))
    }

V_exact = np.array([
    [causal_value_function(coalition, exact_sampler, psi, N=8) for psi in basis] for coalition in coalitions
])
phi_exact = causal_shapely_values(np.array([1, 1, 1]), exact_sampler, f, N=8)
effect_exact = causal_intervention_effect({"X1": 1}, {"X2": 1}, exact_sampler, f, N=8)

assert np.linalg.matrix_rank(V_exact) == V_exact.shape[0]
assert np.allclose(phi_exact, 0)
assert np.isclose(effect_exact, 1)

print("Exact-expectation Shapley values:", phi_exact)
print("Exact causal intervention effect:", effect_exact)
