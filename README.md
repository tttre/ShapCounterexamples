# SHAP Counterexamples

Controlled experiments on SHAP and model dependence.

Install project dependencies (including pandas) from the repository root:

```sh
uv sync --locked
```

Run this again after pulling dependency changes. Commit both `pyproject.toml` and
`uv.lock` when adding dependencies so collaborators use the same resolved versions.

## Repository layout

- `SHAP Counterexamples/src/` and `SHAP Counterexamples/evaluation/`: our journal implementations and experiments.
- [`SHAP Counterexamples/experiment_recreation/`](<SHAP Counterexamples/experiment_recreation/README.md>):
  the adapted SHAP blog subscriber-retention example, including its notebook,
  simulator, dataset wrapper, and XGBoost predictor.
- `SHAP Counterexamples/caches/`: the existing journal data and MLP model caches.

The environment and dependency lockfile are shared at the repository root.
