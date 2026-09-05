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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

# Metrics
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


true_early_stopping = results_df["actual_best"] > results_df["best_so_far"]
predicted_early_stopping = np.ones(len(results_df), dtype=bool)

cm = confusion_matrix(true_early_stopping, predicted_early_stopping)

print("Accuracy :", accuracy_score(true_early_stopping, predicted_early_stopping))
print("Precision:", precision_score(true_early_stopping, predicted_early_stopping))
print("Recall   :", recall_score(true_early_stopping, predicted_early_stopping))
print("F1       :", f1_score(true_early_stopping, predicted_early_stopping))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Keep", "Prune"]
)
disp.plot(cmap=plt.cm.Blues)
plt.show()

loss = results_df.loc[
    results_df["actual_best"] <= results_df["best_so_far"],
    "best_so_far"
] - results_df.loc[
    results_df["actual_best"] <= results_df["best_so_far"],
    "actual_best"
]

print(loss.describe())

# Scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(
    results_df["actual_best"],
    results_df["predicted_best"],
)

mn = min(
    results_df["actual_best"].min(),
    results_df["predicted_best"].min(),
)

mx = max(
    results_df["actual_best"].max(),
    results_df["predicted_best"].max(),
)

plt.plot([mn, mx], [mn, mx], "r--", label="Ideal prediction")

plt.xlabel("Actual best MAE")
plt.ylabel("Predicted best MAE")
plt.title("Prediction quality of curve extrapolation")
plt.legend()
plt.tight_layout()
plt.show()