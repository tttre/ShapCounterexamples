### -----------------------------------------------------------------
### Dataset Generation using different definitions and feature values
### -----------------------------------------------------------------

import numpy as np

### Returns all possible combinations of n bits
def uniform_bit_generator(n_features):
    """
    n_features: number of bits (features)
    """
    return np.array(np.meshgrid(*[[0, 1]] * n_features)).T.reshape(-1, n_features)

### Bernoulli distribution with parameter p for a n_feature-dimensional vector, repeated N times
def bernoulli_bit_generator(n_features, p, N):
    """
    n_features: number of bits (features)
    p: probability of bit 1
    N: number of samples
    """
    rng = np.random.default_rng()
    return rng.binomial(n=1, p=p, size=(N, n_features))

