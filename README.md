# SHAP Counterexamples
### Research Question and Goal

This project combines theoretical analysis and numerical experiments to investigate the relationship between SHAP feature attributions, model dependence, and causal intervention effects.

> Given an SCM (Structural Causal Model), a class of prediction functions and an input,
> when does a causal SHAP attribution vector contain enough information to determine
> the effect of a specified intervention on the expected model output?

We're searching for conditions on the background distribution, prediction function family and causal structure that answer the research question.

### Motivation
Imagine a classifier model is trained on a company's sales data such that it learns the following association: "Customers receiving large discounts often cancel". Under the collected data, SHAP may then show that those discounts contribute towards predicting cancellation, from which the company's director concludes that discounts should be removed.

As the company may offer its largest discounts to customers who are already likely to cancel, this decision may be wrong. Consequently, discounts can be associated with cancellation even when offering a discount improves retention. This motivates distinguishing explanations of predictions from evidence about interventions.

### Running the Examples

With Python 3.12+ and uv installed, run these commands from the repository root:

```sh
uv sync --locked
uv run python "SHAP Counterexamples/evaluation/experiments/recoverability.py"
uv run python "SHAP Counterexamples/evaluation/experiments/recoverability_counterexample.py"
```

The examples generate their own data. The first prints matching direct and reconstructed effects of approximately `-1.26638`. The second demonstrates both a nonrecoverable contrast and a recoverable contrast, followed by a check using all eight binary states.

### Repo Overview
##### Data Generation
Synthetic data is generated using sampler functions defined in "samplers.py". The causal samplers contain the SCM and generate observations from random noise.

The optional "fixed" dictionary implements interventions by replacing the underlying data-generating mechanisms of random variables in "fixed" with constant assignments. This allows downstream changes to propagate through the causal chain.

Example prediction functions are provided in "score_functions.py".

##### SHAP Computation
We differentiate between three different SHAP definitions:

- Conditional SHAP: Conditions on observed feature values.
- Interventional SHAP: Fixes the features in each coalition to their values in the explanation input, while averaging over the joint background distribution of the remaining features.
- Causal SHAP: Intervenes on selected RVs in the SCM and allows for the propagation of causal effects downstream.

Each definition is implemented within "shapely_values.py" and demonstrated in "test_shap_extensions.py".

For conditional SHAP with continuous features, conditional expectations of model outputs are approximated using k nearest neighbour averaging. Samples are matched on conditioned discrete features and selected by proximity in the conditioned continuous features (demonstrated in "test_knn_shapely.py").

##### Causal-Graph Analysis - WIP
The SCM module represents causal graphs, implements graph-based coalition reduction to irreducible sets, and computes the SCM's Shapley weights matrix based on a graphical description.

The Shapley weights matrix is a first attempt to analyze under which conditions Shapley values retain enough information to recover causal intervention effects and when additional information is needed.

### An Example
For a fixed input and prediction-function basis, let `V` contain the expected basis-function values under each coalition intervention, and let `W` contain the Shapley weights. Then `A = W @ V` maps predictor coefficients to causal SHAP values.

For an intervention contrast represented by `r`, if **V has full row rank**, its effect can be reconstructed from causal SHAP values for every predictor in the basis span exactly when `r` belongs to `col(W.T)`. Solving `W.T @ lambdas = r` gives the reconstruction `lambdas @ phi`.

In "recoverability.py", the contrast from `do(X1=1)` to `do(X2=1)` is recoverable. Direct and reconstructed estimates agree because they use the same sampled coalition expectations.

In "recoverability_counterexample.py", three independent Bernoulli(1/2) features and the predictor `f(X) = (X1 - X2) * (4 * X3 - 3)` give causal SHAP values `(0, 0, 0)` at `(1, 1, 1)`, but the effect from `do(X1=1)` to `do(X2=1)` is `1`. By linearity, `-f` has the same attributions and effect `-1`. Therefore, the attribution vector alone cannot determine this effect. The contrast from no intervention to fixing all three features is recoverable.

The sampled results use a fixed seed and approximate the population expectations. The final enumeration checks full row rank and the counterexample using exact expectations; floating-point arithmetic may leave rounding errors near zero.

### Checks

Run the existing function checks from the repository root:

```sh
uv run python "SHAP Counterexamples/evaluation/function tests/test_shapely.py"
uv run python "SHAP Counterexamples/evaluation/function tests/test_decomposition_shapely.py"
uv run python "SHAP Counterexamples/evaluation/function tests/test_knn_shapely.py"
uv run python "SHAP Counterexamples/evaluation/function tests/test_shap_extensions.py"
uv run python "SHAP Counterexamples/evaluation/experiments/theorem41.py"
```

### Related Example

The adapted subscriber-retention example and its source attribution are documented in [experiment_recreation/README.md](SHAP%20Counterexamples/experiment_recreation/README.md).
