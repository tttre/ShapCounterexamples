"""Subscriber-retention simulator based on the Microsoft/SHAP example.

Source:
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html

The equations and random-draw order follow the example. A local RandomState
avoids changing NumPy's global random state. The returned dictionary also keeps
the renewal probability, which the original notebook replaces with a binary label.
"""

import numpy as np



OBSERVED_FEATURES = (
    "Sales calls", "Interactions", "Economy", "Last upgrade",
    "Discount", "Monthly usage", "Ad spend", "Bugs reported",
)
HIDDEN_FEATURES = ("Product need", "Bugs faced")


def retention_generator(n=10_000, fixed=None, seed=0):
    """Generate observations or population interventions in the retention model.

    fixed maps feature names to finite scalar values, e.g. {"Discount": 0.2}.
    Each override is applied before descendants are generated. Hidden features
    and outcome columns are returned for evaluation, not as predictor inputs.

    Reusing a seed reproduces the population-intervention procedure. It does
    not guarantee the same exogenous noise for each individual after arbitrary
    interventions, because Poisson sampling can consume different random draws.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")
    fixed = {} if fixed is None else dict(fixed)
    unknown = set(fixed) - set(OBSERVED_FEATURES + HIDDEN_FEATURES)
    if unknown:
        raise ValueError(f"Unknown intervention features: {sorted(unknown)}")
    for value in fixed.values():
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError("Intervention values must be finite scalars.")

    rng = np.random.RandomState(seed)
    data = {}

    def node(name, natural_value):
        data[name] = (np.full(n, fixed[name]) if name in fixed
                      else np.asarray(natural_value))
        return data[name]

    def sigmoid(value):
        return 1.0 / (1.0 + np.exp(-value))

    sales = node("Sales calls", rng.uniform(0, 4, n).round())
    interactions = node("Interactions", sales + rng.poisson(0.2, n))
    economy = node("Economy", rng.uniform(0, 1, n))
    upgrade = node("Last upgrade", rng.uniform(0, 20, n))
    need = node("Product need", 0.1 * sales + rng.normal(0, 1, n))
    discount = node(
        "Discount", 0.25 * (1.0 - sigmoid(need) + rng.uniform(0, 1, n))
    )
    usage = node("Monthly usage", sigmoid(0.3 * need + rng.normal(0, 1, n)))
    # Preserve the original draw expression and its seeded ordering.
    node("Ad spend", usage * rng.uniform(0.99, 0.9, n)
         + (upgrade < 1) + (upgrade < 2))
    bugs = node("Bugs faced", np.array([rng.poisson(2 * u) for u in usage]))
    node("Bugs reported", (bugs * sigmoid(need)).round())

    probability = sigmoid(7 * (
        0.18 * need + 0.08 * usage + 0.1 * economy + 0.05 * discount
        + 0.05 * rng.normal(0, 1, n) + 0.05 * (1.0 - bugs / 20)
        + 0.005 * sales + 0.015 * interactions
        + 0.1 / (upgrade / 4 + 0.25) - 0.45
    ))
    data["Renewal probability"] = probability
    data["Did renew"] = rng.binomial(1, probability)
    return data
