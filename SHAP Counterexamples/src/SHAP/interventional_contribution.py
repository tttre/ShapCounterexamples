import numpy as np
from .Utils.value_functions import causal_value_function

def interventional_contribution(n_features, func):
    return np.abs(func(np.eye(n_features)) - func(np.zeros((1, n_features)))[0])

def causal_intervention_effect(fixed_before, fixed_after, sampler, func, N=10_000, seed=0):
    """Estimate the change in mean output from fixed_before to fixed_after.

    Each intervention maps feature names to fixed values. Use {} for no intervention.
    """
    before = causal_value_function(fixed_before, sampler, func, N, seed)
    after = causal_value_function(fixed_after, sampler, func, N, seed)
    return after - before

def feature_change_deltas(X, func, targets):
    """Return signed changes, one per input row and feature.

    X:       array of shape (n_samples, n_features)
    func:    callable returning one prediction per row
    targets: replacement values, broadcastable to X.shape
    """
    X = np.asarray(X, dtype=float)
    targets = np.broadcast_to(
        np.asarray(targets, dtype=float), X.shape
    )

    before = np.asarray(func(X)).reshape(-1)
    deltas = np.empty(X.shape, dtype=float)

    for i in range(X.shape[1]):
        changed = X.copy()
        changed[:, i] = targets[:, i]
        after = np.asarray(func(changed)).reshape(-1)
        deltas[:, i] = after - before

    return deltas
