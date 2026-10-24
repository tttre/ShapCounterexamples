### ----------------------------------------------------
### Value functions with their respective simplifcations 
### ----------------------------------------------------

import numpy as np

### Mean Value Function with emprical estimation of the conditional expectation from the dataset
def mean_value_function(fixed_features, fixed_values, data, func):
    matches = np.all(data[:, fixed_features] == fixed_values,axis=1) #Creates a mask
    return np.asarray(func(data[matches])).mean().item()

### k Nearest Neighbour Value Function pool is either 
### 1) a large generated sample from the known distribution of the features
### 2) the dataset, if the distribution is unknown
def kNN_value_function(fixed_features, fixed_values, pool, func, continuous_features, k):
    fixed = dict(zip(fixed_features, fixed_values))
    names = list(pool)

    # Knowing every feature gives f(x) directly.
    if len(fixed) == len(names):
        point = {
            name: np.array([fixed[i]])
            for i, name in enumerate(names)
        }
        return np.asarray(func(point)).mean().item()

    
    N = len(pool[names[0]])

    matches = np.ones(N, dtype=bool) # Discrete feature mask, being true, if an input sample's feature matches the fixed value
    distance = np.zeros(N) # Continuous feature distance, being the squared distance of a sample's continuous features to the fixed values

    # Iterate over every fixed feature, and update the matches and distance arrays accordingly
    for i, value in fixed.items():
        column = pool[names[i]]

        if i in continuous_features:
            scale = float(np.std(column)) or 1.0
            distance += ((column - value) / scale) ** 2
        else:
            matches &= column == value


    # 1)------
    indices = np.flatnonzero(matches) #Compresses index vector down to only the indices of the samples that match the fixed discrete values
    # --------
    if indices.size == 0:
        raise ValueError("No matching discrete values in the sample pool.")

    if any(i in continuous_features for i in fixed):
        # 2)------
        indices = indices[np.argsort(distance[indices])[:k]]
        # --------

    # 1) and 2) together say, for the samples that match the fixed discrete values AND that are closest to the fixed continuous values
    # take now the first k of these and mean.
    selected = {name: column[indices] for name, column in pool.items()}
    return np.asarray(func(selected)).mean().item()