### -------------------------------------------------------------------------------------------
### Generate .npz file containing dataset, specifically {data, scores}, where data is generated
### for some generator, and scores is generated for some score function.
### -------------------------------------------------------------------------------------------

import numpy as np
from .Utils.score_functions import nonlinear_score
from .Utils.generators import uniform_bit_generator, bernoulli_bit_generator, hidden_bit_generator

from pathlib import Path

### Generate Dataset based on a generator and a score function, and save it to a .npz file.
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

### Generate and save datasets
def main():
    n_features = 3
    N_samples = 1000
    seed = 0
    coefficients = (1, 1, 1)
    p3_values = np.linspace(0.1, 0.9, 30)
    lambda_values = np.linspace(0.0, 1.0, 30)

    cache_dir = (
        Path(__file__).resolve().parents[2]
        / "caches" / "data_cache" / "experiments_21_10_26" / "experiment1"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    for p_index, p3 in enumerate(p3_values):
        p = np.array([0.5, 0.5, p3])
        sample_seed = seed + p_index
        # Reuse inputs across lambda values to isolate changes in the score.
        X = np.asarray(
            bernoulli_bit_generator(n_features, p, N_samples, seed=sample_seed),
            dtype=np.float32,
        )
        for lambda_index, lambda_value in enumerate(lambda_values):
            scores = nonlinear_score(X, lambda_value, *coefficients).astype(np.float32)
            output_path = cache_dir / (
                f"p{p_index:02d}_{p3:.4f}_lambda{lambda_index:02d}_{lambda_value:.4f}.npz"
            )
            # p stores all three probabilities; the filename reports p3.
            np.savez(
                output_path,
                X=X,
                scores=scores,
                p=p,
                lambda_value=lambda_value,
                coefficients=np.asarray(coefficients),
                seed=sample_seed,
                N_samples=N_samples,
            )

    print(f"Saved {len(p3_values) * len(lambda_values)} datasets to: {cache_dir}")


if __name__ == "__main__":
    main()
