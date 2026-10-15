### -------------------------------------------------------------------
### Exact conditional Shapely value computation used for small datasets
### -------------------------------------------------------------------

import numpy as np
from itertools import combinations
from math import comb
from .Utils.value_functions import mean_value_function

### Determine the Shapley Value for a given feature based on the value function
def feature_shapely(x, X, feature, func):
    """
    x: The input sample for which the Shapely values should be computed
    X: The dataset used for conditional shapely
    feature: The index of the feature which should be held constant
    func: The explanable function
    """
    d = X.shape[1]
    others = [i for i in range(d) if i != feature]
    phi = 0.0

    for size in range(d):
        for coalition in combinations(others, size):
            S = list(coalition)
            S_with_feature = S + [feature]

            value_without = mean_value_function(S, x[S], X, func)
            value_with = mean_value_function(S_with_feature, x[S_with_feature], X, func)

            weight = 1 / (d * comb(d - 1, size))

            phi += weight * (value_with - value_without)

    return phi

### Compute conditional Shapely values exact for some trained model
def exact_shapely(x, X, func):
    """
    x: The input sample for which the Shapely values should be computed
    X: The dataset used for conditional shapely
    feature: The index of the feature which should be held constant
    func: The explanable function
    """
    d = X.shape[1]
    phi = np.zeros(d)

    for feature in range(d):
        phi[feature] = feature_shapely(x, X, feature, func)

    return phi