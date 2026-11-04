# SHAP subscriber-retention recreation

This folder contains the local adaptation of the SHAP documentation article
[Be careful when interpreting predictive models in search of causal insights](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html),
by Eleanor Dillon, Jacob LaRiviere, Scott Lundberg, Jonathan Roth, and Vasilis Syrgkanis.
The retention implementation entered this repository in
[commit 56f109a](https://github.com/tttre/ShapCounterexamples/commit/56f109a065cbc46f77cc55434bdf29a7e39a73e9).
These are locally adapted files, rather than an unmodified copy of the article.

## Contents

- `Retention.ipynb`: dataset generation, XGBoost training, and TreeSHAP analysis.
- `Simulator/retention.py`: the data-generating model, including population interventions through `fixed`.
- `Predictors/retention_xgboost.py`: the XGBoost training helper.
- `synthesis.py`: the observed-feature dataset wrapper formerly in the shared data module.

The journal's custom SHAP implementations, binary and housing experiments, generic
feature-change helpers, and MLP code remain in `../src/`.
The existing `../caches/model_cache/model.pt` belongs to the MLP workflow and predates
the retention commit; it is not a retention-model checkpoint.

## Running

Use the existing shared environment. From the repository root:

```sh
uv sync --locked --group dev
uv run --group dev jupyter notebook "SHAP Counterexamples/experiment_recreation/Retention.ipynb"
```

Select the repository's Python environment and run the notebook from top to bottom.
Its first code cell locates the project directory when the working directory is the
repository root or any directory inside it. Saved outputs from the earlier run are
preserved. Training parameters and simulation equations are unchanged.

From Python launched inside `SHAP Counterexamples/`:

```python
from experiment_recreation.synthesis import generate_retention_dataset
from experiment_recreation.Simulator.retention import retention_generator
from experiment_recreation.Predictors.retention_xgboost import fit_retention_xgboost
```

Dependencies remain declared in the repository-root `pyproject.toml` and `uv.lock`.
