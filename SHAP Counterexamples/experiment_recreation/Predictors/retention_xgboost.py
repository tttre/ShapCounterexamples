import xgboost as xgb
from sklearn.model_selection import train_test_split


def fit_retention_xgboost(
    X,
    y,
    seed=0,
    validation_size=0.25,
    num_boost_round=200_000,
    early_stopping_rounds=20,
):

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=seed,
        stratify=y,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:logistic",
        "eval_metric": "rmse",
        "eta": 0.001,
        "subsample": 0.5,
        "max_depth": 2,
        "seed": seed,
        "nthread": 4,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    # Retain the trees through the best validation iteration.
    return model[:model.best_iteration + 1]