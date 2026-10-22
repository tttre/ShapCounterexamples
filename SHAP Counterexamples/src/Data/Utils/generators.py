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
def bernoulli_bit_generator(n_features, p, N, seed=None):
    """
    n_features: number of bits (features)
    p: probability of bit 1
    N: number of samples
    """
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=p, size=(N, n_features))

### Observed noisy copies of a shared hidden Bernoulli feature
def hidden_bit_generator(n_features, q, N, seed=None):
    """
    n_features: number of observed features
    q: probability of flipping the hidden bit independently for each feature
    N: number of samples
    """
    rng = np.random.default_rng(seed)
    H = rng.binomial(n=1, p=0.5, size=(N, 1))
    noise = rng.binomial(n=1, p=q, size=(N, n_features))
    return H ^ noise

