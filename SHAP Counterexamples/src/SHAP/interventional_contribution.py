import numpy as np

def interventional_contribution(n_features, func):
    return np.abs(func(np.eye(n_features)) - func(np.zeros((1, n_features)))[0])
