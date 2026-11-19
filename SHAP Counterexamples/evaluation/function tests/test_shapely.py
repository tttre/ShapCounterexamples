### ----------------------------------------------------------------
### Temporary testing file for exact Shapley implementation
### ----------------------------------------------------------------

import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.Data.Utils.score_functions import binary_score
from src.Data.Utils.generators import uniform_bit_generator
from src.SHAP.shapely_values import shapely_values


def feature_importance(X):
    shap_values = np.array([
        shapely_values(x, X, binary_score)
        for x in X
    ])

    return np.abs(shap_values).mean(axis=0)

# Generate all three-bit inputs.
X = uniform_bit_generator(3)

print("Feature Importance:", feature_importance(X))
