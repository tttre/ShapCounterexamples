from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Data.Utils.samplers import causal_house_sampler, conditional_house_sampler
from src.Data.Utils.score_functions import house_price_score
from src.SHAP.shapely_values import kNN_shapely_values, interventional_shapely_values, causal_shapely_values


pool = conditional_house_sampler(10000, np.random.default_rng(0))

# Feature order: bedrooms, area, location
X = [
    [2, 80, "suburb"],
    [3, 100, "city"],
    [5, 150, "city"],
]

for x in X:
    x = np.asarray(x, dtype=object)
    phi_conditional = kNN_shapely_values(x, pool, house_price_score, continuous_features=[0, 1], k=200)
    phi_interventional = interventional_shapely_values(x, pool, house_price_score)
    phi_causal = causal_shapely_values(x, causal_house_sampler, house_price_score, N=10000)
    print("----------------------------------------------------------------")
    print("SHAP Values for input:", x)
    print("Conditional SHAP Values:")
    print(phi_conditional)
    print("Interventional SHAP Values:")
    print(phi_interventional)
    print("Causal SHAP Values:")
    print(phi_causal)
    print("----------------------------------------------------------------")