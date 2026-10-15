### ----------------------------------------------------------------
### Temporary testing file for exact Shapley implementation
### ----------------------------------------------------------------

import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Data.Utils.score_functions import single_feature_binary_score
from SHAP.exact_shapely import exact_shapely


def feature_importance(X):
    shap_values = np.array([
        exact_shapely(x, X, single_feature_binary_score)
        for x in X
    ])

    return np.abs(shap_values).mean(axis=0)

caches = PROJECT_ROOT / "caches"

# Load inputs.
with np.load(caches / "data_cache" / "bit_data.npz") as data:
    X = data["X"]

print("Feature Importance:", feature_importance(X))
