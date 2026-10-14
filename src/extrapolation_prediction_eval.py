import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split



TEST_SIZE = 0.2


def train_xgboost(num_boost_round, params, dtrain, dval):
    evals_result = {}

    xgb.train(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        evals_result=evals_result,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )

    return evals_result


df = pd.read_csv(
    "curve-extrapolation-based-stopping-criteria-within-the-BO/data/winequality-red.csv"
)

y = df.quality
X = df.iloc[:, :-1]

early_stopped_trials = pd.read_csv(
    "curve-extrapolation-based-stopping-criteria-within-the-BO/results/early_stopped_trials.csv"
)


results = []

for _, row in early_stopped_trials.iterrows():

    seed = int(row["seed"])
    predicted_best = row["predicted_best"]

    # Recover parameter dictionary
    params = literal_eval(row["params"])

    # Extract num_boost_round
    num_boost_round = params.pop("num_boost_round")

    # Restore fixed parameters
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "mae"
    params["seed"] = seed

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=TEST_SIZE,
        random_state=seed,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals_result = train_xgboost(
        num_boost_round,
        params,
        dtrain,
        dval,
    )

    actual_best = min(evals_result["val"]["mae"])
    error = predicted_best - actual_best

    results.append(
        {
            "seed": seed,
            "predicted_best": predicted_best,
            "actual_best": actual_best,
            "error": error,
            "absolute_error": abs(error),
            "num_boost_round": num_boost_round,
            "best_so_far": row["best_so_far"]
        }
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "curve-extrapolation-based-stopping-criteria-within-the-BO/results/prediction_validation.csv",
    index=False,
)

errors = results_df["error"]
abs_errors = results_df["absolute_error"]



prediction_mae = abs_errors.mean()
bias = errors.mean()
rmse = np.sqrt((errors**2).mean())
correlation = np.corrcoef(
    results_df["predicted_best"],
    results_df["actual_best"],
)[0, 1]



print(f"Prediction MAE : {prediction_mae:.6f}")
print(f"Bias           : {bias:.6f}")
print(f"RMSE           : {rmse:.6f}")
print(f"Correlation    : {correlation:.4f}")