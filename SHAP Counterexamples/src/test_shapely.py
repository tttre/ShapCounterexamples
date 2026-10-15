import numpy as np
import torch

from pathlib import Path

from NN.MLP.Utils.prediction import predict
from NN.MLP.model import MLP
from SHAP.exact_shapely import exact_shapely


def feature_importance(model, X):
    predict_fn = lambda data: predict(model, data)

    shap_values = np.array([
        exact_shapely(x, X, predict_fn)
        for x in X
    ])

    return np.abs(shap_values).mean(axis=0)

caches = Path("caches")

# Load inputs.
with np.load(caches / "data_cache" / "bit_data.npz") as data:
    X = data["X"]

# Recreate the network and load its trained weights.
model = MLP(n_features=X.shape[1])
model.load_state_dict(torch.load(
    caches / "model_cache" / "model.pt",
    map_location="cpu",
    weights_only=True,
))
model.eval()

print("Feature Importance:", feature_importance(model, X))
