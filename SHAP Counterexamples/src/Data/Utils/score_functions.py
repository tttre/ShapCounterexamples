### -----------------------------------------------------------
### Score Functions used to generate the target of some dataset
### -----------------------------------------------------------

import numpy as np

def polynomial_score(X):
        return 0.1*X[:, 0] + 0.2*X[:, 1] + 0.3*X[:, 2]

def binary_score(X):
        return np.logical_or( X[:, 0], np.logical_and(X[:, 1], X[:, 2]))