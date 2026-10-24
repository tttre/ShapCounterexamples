### -------------------------------------------------------------------------------------------
### Score Functions used to generate the target of some dataset (supports only numeric features
### and assumes descriptive sampler)
### -------------------------------------------------------------------------------------------

import numpy as np

### -------------------------------------------------------------------------------------------
### Single Feature Scores
def binary_score(X):
    columns = list(X.values()) if isinstance(X, dict) else X.T
    return columns[0]
### -------------------------------------------------------------------------------------------

### -------------------------------------------------------------------------------------------
### Three Feature Score
def polynomial_score(X):
    columns = list(X.values()) if isinstance(X, dict) else X.T
    return 0.1*columns[0] + 0.2*columns[1] + 0.3*columns[2]

def nonlinear_score(X,l,a,b,c):
    columns = list(X.values()) if isinstance(X, dict) else X.T
    return a*columns[0] + b*columns[1] + c*columns[2] + l*columns[0]*columns[1]*columns[2]
### -------------------------------------------------------------------------------------------
