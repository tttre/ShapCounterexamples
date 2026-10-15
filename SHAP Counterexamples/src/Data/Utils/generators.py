import numpy as np

### Returns all possible combinations of n bits.
def bit_generator(n):
    return np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)