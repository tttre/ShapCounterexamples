### -----------------------------------------------------------------
### Dataset Generation using different definitions and feature values
### -----------------------------------------------------------------

import numpy as np
### -----------------------------------------------------------------
### Primitive generators for synthetic datasets
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
### -----------------------------------------------------------------

### -----------------------------------------------------------------
### Feature generators for synthetic datasets
import numpy as np


def binary_feature_generator(p, N, seed=None):
    """Generate N binary values with probability p of being 1."""
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=p, size=N)


def gaussian_feature_generator(mean, std, N, seed=None):
    """Generate N Gaussian values with the given mean and standard deviation."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=N)


def uniform_feature_generator(low, high, N, seed=None):
    """Generate N uniformly distributed values between low and high."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=N)


def count_feature_generator(N, lam=1.0, seed=None):
    """Generate N Poisson counts with expected count lam."""
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=lam, size=N)


def categorical_feature_generator(categories, N, seed=None):
    """Generate N values uniformly from a single list of categories."""
    rng = np.random.default_rng(seed)
    return rng.choice(categories, size=N)
### -----------------------------------------------------------------