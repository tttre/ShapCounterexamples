### ------------------------------------------------
### Decomposition of shapely values into A_i and B_i
### ------------------------------------------------

import numpy as np
from itertools import combinations
from math import comb
from .Utils.value_functions import mean_value_function

### Determine the decomposed Shapley value for a given feature based on the value function
def feature_decomposition_shapely(x, X, feature, func):
    """
    x: The input sample for which the Shapely values should be computed
    X: The dataset used for conditional shapely
    feature: The index of the feature which should be held constant
    func: The explanable function
    """
    d = X.shape[1]
    others = [i for i in range(d) if i != feature]
    phiA = 0.0
    phiB = 0.0

    for size in range(d):
        for coalition in combinations(others, size):
            S = list(coalition)
            S_with_feature = S + [feature]

            X_fixed = X.copy()
            X_fixed[:, feature] = x[feature]

            A_i = mean_value_function(S, x[S], X_fixed, func) - mean_value_function(S, x[S], X, func)
            B_i = mean_value_function(S_with_feature, x[S_with_feature], X, func) - mean_value_function(S, x[S], X_fixed, func)

            weight = 1 / (d * comb(d - 1, size))

            phiA += weight * A_i
            phiB += weight * B_i

    return phiA, phiB

def shapely_values_decomposition(x, X, func):
    """
    x: The input sample for which the Shapely values should be computed
    X: The dataset used for conditional shapely
    feature: The index of the feature which should be held constant
    func: The explanable function
    """
    d = X.shape[1]
    phiA = np.zeros(d)
    phiB = np.zeros(d)

    for feature in range(d):
        phiA[feature], phiB[feature] = feature_decomposition_shapely(x, X, feature, func)

    return phiA, phiB