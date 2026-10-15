### -----------------------------------------------------------
### Score Functions used to generate the target of some dataset
### -----------------------------------------------------------

import numpy as np

def polynomial_score(X):
        return 0.1*X[:, 0] + 0.2*X[:, 1] + 0.3*X[:, 2]

def binary_score(X):
        return X[:,0] or (X[:,1] and X[:,2])