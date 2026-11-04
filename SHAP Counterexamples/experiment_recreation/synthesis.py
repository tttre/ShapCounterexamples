"""Dataset wrapper for the adapted SHAP subscriber-retention example."""

import pandas as pd

from .Simulator.retention import OBSERVED_FEATURES, retention_generator


def generate_retention_dataset(n=10_000, seed = 0):
    """Generate observed retention features and a binary renewal target.

    Returns a DataFrame containing the eight observed features and a
    'score' column, following the existing dataset convention.
    """
    simulation = retention_generator(n=n, seed=seed)

    dataset = pd.DataFrame({
        name: simulation[name]
        for name in OBSERVED_FEATURES
    })

    dataset["score"] = simulation["Did renew"]

    return dataset
