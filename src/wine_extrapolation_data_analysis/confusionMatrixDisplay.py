import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

early_stopped_trials = pd.read_csv(
    "curve-extrapolation-based-stopping-criteria-within-the-BO/results/wine_early_stopped_trials.csv"
)

results_df = pd.read_csv(
    "curve-extrapolation-based-stopping-criteria-within-the-BO/results/wine_prediction_validation.csv"
)

total_boost_rounds = results_df["num_boost_round"].sum()
percentage_saved = sum(num_boost * 0.2 for n, num_boost in enumerate(early_stopped_trials["num_boost_round"].to_numpy()) if early_stopped_trials["early_stopped"][n].astype(bool)) / total_boost_rounds * 100
print(f"Total boost rounds: {total_boost_rounds}")
print(f"Percentage of boost rounds saved: {100 - percentage_saved:.2f}%")

true_early_stopping = pd.concat([
    results_df["actual_best"] > results_df["best_so_far"],
    pd.Series([False] * 15)
]).to_numpy()

predicted_early_stopping = pd.concat([
    early_stopped_trials["early_stopped"].astype(bool),
    pd.Series([False] * 15)
]).to_numpy()

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

df_unido = results_df

df_unido["early_stopped"] = early_stopped_trials["early_stopped"]

absolute_error = (
    df_unido.loc[df_unido["early_stopped"], "predicted_best"]
    -
    df_unido.loc[df_unido["early_stopped"], "actual_best"]
).abs()

signed_error = (
    df_unido.loc[df_unido["early_stopped"], "predicted_best"]
    -
    df_unido.loc[df_unido["early_stopped"], "actual_best"]
)

print(absolute_error.describe())

print(signed_error.describe())

df_unido_early_stopped = df_unido[df_unido["early_stopped"] == True]

print(df_unido_early_stopped["early_stopped"])


# Scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(
    df_unido_early_stopped["actual_best"],
    df_unido_early_stopped["predicted_best"],
)

mn = min(
    df_unido_early_stopped["actual_best"].min(),
    df_unido_early_stopped["predicted_best"].min(),
)

mx = max(
    df_unido_early_stopped["actual_best"].max(),
    df_unido_early_stopped["predicted_best"].max(),
)

plt.plot([mn, mx], [mn, mx], "r--", label="Ideal prediction")

plt.xlabel("Actual best MAE")
plt.ylabel("Predicted best MAE")
plt.title("Prediction quality of curve extrapolation")
plt.legend()
plt.tight_layout()
plt.show()