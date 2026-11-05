from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Data.Utils.samplers import causal_house_sampler_2
from src.Data.Utils.score_functions import house_price_score_2
from src.SHAP.shapely_values import causal_shapely_values
from src.SHAP.interventional_contribution import causal_intervention_effect


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
    print("----------------------------------------------------------------")
