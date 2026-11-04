### -------------------------------------------------------------------
### Exact conditional Shapely value computation used for small datasets
### -------------------------------------------------------------------

import numpy as np
from itertools import combinations
from math import comb
from .Utils.value_functions import mean_value_function, kNN_value_function, causal_value_function
from ..Data.Utils.samplers import temp_sampler

### -------------------------------------------------------------------
### Conditional SHAP

### Determine the Shapely value for a given feature based on a known distribution programmed in some 
### distribution sampler
def kNN_feature_shapely(x, pool, feature, func, continuous_features, k=200):
    d = len(pool)
    others = [i for i in range(d) if i != feature]
    phi = 0.0

    for size in range(d):
        for coalition in combinations(others, size):
            S = list(coalition)
            S_with_feature = S + [feature]

            value_without = kNN_value_function(S, x[S], pool, func, continuous_features, k)
            value_with = kNN_value_function(S_with_feature, x[S_with_feature], pool, func, continuous_features, k)

            weight = 1 / (d * comb(d - 1, size))

            phi += weight * (value_with - value_without)

    return phi

### Determine the Shapley value for a given feature based on the value function
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

### Compute conditional Shapely values exact for some model func
def shapely_values(x, X, func):
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

### Compute conditional Shapely values exact for some model func using kNN value function as 
### conditional expectation estimate
def kNN_shapely_values(x, pool, func, continuous_features, k=200):
    """
    x: The input sample for which the Shapely values should be computed
    pool: The dataset used for conditional shapely
    func: The explanable function
    continuous_features: List of indices of continuous features
    k: Number of nearest neighbors to consider
    """
    d = len(pool)
    phi = np.zeros(d)

    for feature in range(d):
        phi[feature] = kNN_feature_shapely(x, pool, feature, func, continuous_features, k)

    return phi
### -------------------------------------------------------------------

### -------------------------------------------------------------------
### Causal SHAP
### Compute causal Shapely values for some model func using sampled causal expectations
def causal_shapely_values(x, sampler, func, N, seed=0):
    """
    x: The input sample for which the Shapely values should be computed
    sampler: The causal sampler used to generate intervened samples
    func: The explainable function
    """
    # Generate one sample to obtain the feature names
    pool = sampler(1, np.random.default_rng(seed), fixed={})
    feature_names = list(pool)

    d = len(feature_names)
    phi = np.zeros(d)

    for feature in range(d):
        phi[feature] = causal_feature_shapely(x, sampler, feature, func, feature_names, N, seed)

    return phi
### Determine the causal Shapley value for a given feature based on the value function
def causal_feature_shapely(x, sampler, feature, func, feature_names, N, seed=0):
    """
    x: The input sample for which the Shapely values should be computed
    sampler: The causal sampler used to generate intervened samples
    feature: The index of the feature which should be held constant
    func: The explainable function
    feature_names: The feature names in the same order as x
    """
    d = len(feature_names)
    others = [i for i in range(d) if i != feature]
    phi = 0.0

    for size in range(d):
        for coalition in combinations(others, size):
            S = list(coalition)
            S_with_feature = S + [feature]

            fixed_without = {feature_names[i]: x[i] for i in S}
            fixed_with = {feature_names[i]: x[i] for i in S_with_feature}

            value_without = causal_value_function(fixed_without, sampler, func, N, seed)
            value_with = causal_value_function(fixed_with, sampler, func, N, seed)

            weight = 1 / (d * comb(d - 1, size))

            phi += weight * (value_with - value_without)

    return phi
### -------------------------------------------------------------------


### Population feature importance
def feature_importance(X, func):
    rows, counts = np.unique(X, axis=0, return_counts=True)
    shap_values = np.array([shapely_values(x, X, func) for x in rows])
    return np.average(np.abs(shap_values), axis=0, weights=counts)