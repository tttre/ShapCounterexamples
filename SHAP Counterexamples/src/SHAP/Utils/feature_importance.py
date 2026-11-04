import numpy as np

### Population feature importance
def feature_importance(X, func, shapely_values):
    rows, counts = np.unique(X, axis=0, return_counts=True)
    shap_values = np.array([shapely_values(x, X, func) for x in rows])
    return np.average(np.abs(shap_values), axis=0, weights=counts)