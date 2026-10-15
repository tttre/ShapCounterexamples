import numpy as np

def polynomial_score(X):
        return 0.1*X[:, 0] + 0.2*X[:, 1] + 0.3*X[:, 2]