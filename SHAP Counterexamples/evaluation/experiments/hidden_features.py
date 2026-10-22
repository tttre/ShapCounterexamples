### Hidden feature experiment: conditional importance versus input-change contributions

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.Data.synthesis import generate_hidden_dataset, generate_bernouilli_dataset
from src.Data.Utils.score_functions import binary_score
from src.SHAP.exact_shapely import feature_importance
from src.SHAP.interventional_contribution import interventional_contribution

### Hidden Feature Journal Experiment Reproduction
def experiment1():
    n_features = 2
    N_samples = 10000
    seed = 0
    p_values = np.linspace(0.0, 1.0, 10)

    # f(x) = x1 gives input-change contributions (1, 0).
    contributions = interventional_contribution(n_features, binary_score)
    normalized_contributions = contributions / contributions.sum()
    disagreement_scores = []

    for p in p_values:
        dataset = generate_hidden_dataset(n_features,N_samples,p,binary_score,seed=seed)
        X = dataset["X"]

        # Normalized feature importance
        importance = feature_importance(X, binary_score)
        normalized_importance = importance / importance.sum()
        disagreement_score = 0.5 * np.abs(normalized_importance - normalized_contributions).sum()
        disagreement_scores.append(disagreement_score)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p_values, disagreement_scores, marker="o")
    ax.set(xlabel="Probability p", ylabel="Disagreement score D",
           title=f"Hidden feature experiment (N = {N_samples:,})")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


### Accidental sample correlations in independent two-feature datasets
def experiment2():
    n_features = 2
    N_samples = 100000
    N_datasets = 100
    seed = 0
    score_function = binary_score  # f(x) = x1

    contributions = interventional_contribution(n_features, score_function)
    normalized_contributions = contributions / contributions.sum()
    disagreement_scores = np.zeros(N_datasets)
    correlations = np.zeros(N_datasets)

    for i in range(N_datasets):
        dataset = generate_bernouilli_dataset(
            n_features, N_samples, 0.5, score_function, seed=seed + i
        )
        X = dataset["X"]
        importance = feature_importance(X, score_function)
        normalized_importance = importance / importance.sum()
        disagreement_scores[i] = 0.5 * np.abs(
            normalized_importance - normalized_contributions
        ).sum()
        correlations[i] = np.corrcoef(X, rowvar=False)[0, 1]

    # Each point represents one dataset.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.abs(correlations), disagreement_scores, s=20, alpha=0.7)
    ax.set(xlabel="Absolute sample correlation |r|", ylabel="Disagreement score D",
           title=f"{N_datasets} independent datasets (N = {N_samples:,})")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


### Sweep the correlations of two observed features with their shared hidden bit
def experiment3():
    n_features = 2
    N_samples = 100000
    seed = 0
    score_function = binary_score  # f(x) = x1
    rho1_values = np.linspace(-1.0, 1.0, 21)
    rho2_values = np.linspace(-1.0, 1.0, 21)

    contributions = interventional_contribution(n_features, score_function)
    normalized_contributions = contributions / contributions.sum()
    disagreement_scores = np.zeros((len(rho2_values), len(rho1_values)))

    for row, rho2 in enumerate(rho2_values):
        for col, rho1 in enumerate(rho1_values):
            # Xi = H XOR Ei gives Corr(H, Xi) = 1 - 2*p_i.
            # These are population correlations; H is excluded from the returned X.
            p = (1 - np.array([rho1, rho2])) / 2
            dataset = generate_hidden_dataset(
                n_features, N_samples, p, score_function, seed=seed
            )
            importance = feature_importance(dataset["X"], score_function)
            normalized_importance = importance / importance.sum()
            disagreement_scores[row, col] = 0.5 * np.abs(
                normalized_importance - normalized_contributions
            ).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.pcolormesh(rho1_values, rho2_values, disagreement_scores,
                          shading="nearest", cmap="viridis", vmin=0)
    ax.set(xlabel="Target correlation Corr(H, X1)", ylabel="Target correlation Corr(H, X2)",
           title=f"Shared hidden feature (N = {N_samples:,})",
           xlim=(-1, 1), ylim=(-1, 1),
           xticks=np.linspace(-1, 1, 5), yticks=np.linspace(-1, 1, 5))
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label="Disagreement score D")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment3()
