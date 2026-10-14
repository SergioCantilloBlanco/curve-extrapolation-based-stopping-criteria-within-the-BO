import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv(
    "curve-extrapolation-based-stopping-criteria-within-the-BO/results/extrapolation_results.csv"
)
# Best MAE metrics

mean_mae = results_df["best_mae"].mean()
std_mae = results_df["best_mae"].std()
median_mae = results_df["best_mae"].median()
min_mae = results_df["best_mae"].min()
max_mae = results_df["best_mae"].max()

# Execution time metrics

mean_time = results_df["execution_time"].mean()
std_time = results_df["execution_time"].std()
median_time = results_df["execution_time"].median()
total_time = results_df["execution_time"].sum()

# Early stopping metrics

mean_early_stopped = results_df["early_stopped_trials"].mean()
total_early_stopped = results_df["early_stopped_trials"].sum()

total_trials = results_df["trials"].sum()

early_stopping_rate = (
    total_early_stopped / total_trials
)

# Print results

print("===== EXTRAPOLATION RESULTS =====")

print(f"Mean best MAE          : {mean_mae:.6f}")
print(f"Std best MAE           : {std_mae:.6f}")
print(f"Median best MAE        : {median_mae:.6f}")
print(f"Minimum best MAE       : {min_mae:.6f}")
print(f"Maximum best MAE       : {max_mae:.6f}")

print()

print(f"Mean execution time    : {mean_time:.2f} s")
print(f"Std execution time     : {std_time:.2f} s")
print(f"Median execution time  : {median_time:.2f} s")
print(f"Total execution time   : {total_time:.2f} s")

print()

print(f"Mean early stopped     : {mean_early_stopped:.2f}")
print(f"Total early stopped    : {total_early_stopped}")
print(f"Total trials           : {total_trials}")
print(f"Early stopping rate    : {early_stopping_rate:.2%}")