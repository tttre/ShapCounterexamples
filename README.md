# SHAP Counterexamples

Numerical experiments analyzing when SHAP feature attribution rankings diverge from model dependence and causal intervention. This Projects aims to investigate the following research question:

"When do causal SHAP explanations and causal intervention effects diverge?" or more specific "Given a specified causal model, does a SHAP explanation retain enough information to answer a particular intervention question?"

The project focuses on finding conditions on background distributions and function families in order to derive conditions, when causal SHAP feature attribution alone doesn't cover what an intervention causally changes in a system.

Imagine a classifier model is trained on a company's sales data such that it learns the following causality: "A customer receiving large discounts often cancel". Under the collected data, SHAP may then show that those discounts contribute towards predicting cancellation, from which the company's director concludes that discounts should be removed. This decision can be wrong, especially when the model doesn't "understand" the causal relations between the features and their outputs.