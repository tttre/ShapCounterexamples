### -------------------------------------------------------------------------------------------
### Generate .npz/.csv file containing dataset
### for some generator, (for some sampler) and for some score function
### -------------------------------------------------------------------------------------------

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Preserve relative imports when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.Data"

from .Utils.score_functions import nonlinear_score, binary_score, house_price_score
from .Utils.generators import binary_feature_generator, gaussian_feature_generator, uniform_feature_generator, count_feature_generator, categorical_feature_generator, uniform_bit_generator, bernoulli_bit_generator, hidden_bit_generator
from .Utils.samplers import temp_sampler, conditional_house_sampler

### -------------------------------------------------------------------------------------------
### Primitive generators for synthetic datasets
### Generate Dataset based on a generator and a score function
def generate_uniform_dataset(n_features,score_function):
    """
    n_features: Number of features in the dataset.
    """

    X = np.asarray(uniform_bit_generator(n_features), dtype=np.float32)
    scores = score_function(X).astype(np.float32)
    return {"X": X, "scores": scores}

def generate_bernouilli_dataset(n_features,N_samples,p,score_function,seed=None):
    """
    n_features: Number of features in the dataset.
    N_samples: Number of samples from distribution.
    p: Probability of bit 1 in Bernoulli distribution.
    seed: RNG seed for reproducibility.
    """
    X = np.asarray(bernoulli_bit_generator(n_features,p,N_samples,seed=seed), dtype=np.float32)
    scores = score_function(X).astype(np.float32)
    return {"X": X, "scores": scores}

def generate_hidden_dataset(n_features,N_samples,p,score_function,seed=None):
    """
    n_features: Number of features in the dataset.
    N_samples: Number of samples from distribution.
    p: Probability of bit 1 in Bernoulli distribution.
    seed: RNG seed for reproducibility.
    """
    X = np.asarray(hidden_bit_generator(n_features,p,N_samples,seed=seed), dtype=np.float32)
    scores = score_function(X).astype(np.float32)
    return {"X": X, "scores": scores}
### -------------------------------------------------------------------------------------------

### -------------------------------------------------------------------------------------------
### Generate dataset based on a descriptive sampler (which also allows support for 
### hidden variables) and a score function, returning a pandas DataFrame with features and scores.
def generate_dataset(sampler, score_function, N, seed=None):
    rng = np.random.default_rng(seed)
    features = sampler(N, rng)

    dataset = pd.DataFrame(features)
    dataset["score"] = score_function(features)

    return dataset

### Generate and save datasets
def main():
    dataset = generate_dataset(
        sampler=conditional_house_sampler,
        score_function=house_price_score,
        N=1000,
        seed=0,
    )

    cache_dir = (
        Path(__file__).resolve().parents[2]
        / "caches" / "data_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "house_price_dataset.csv"
    dataset.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
