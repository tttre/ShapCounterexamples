### ----------------------------------------------------
### Value functions with their respective simplifcations 
### ----------------------------------------------------

import numpy as np

### Mean Value Function with emprical estimation of the conditional expectation from the dataset
def mean_value_function(fixed_features, fixed_values, data, func):
    matches = np.all(data[:, fixed_features] == fixed_values,axis=1) #Creates a mask
    return np.asarray(func(data[matches])).mean().item()