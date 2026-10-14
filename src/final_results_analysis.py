# ============================================================
# ANALYSIS OF CURVE EXTRAPOLATION EXPERIMENTS
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import friedmanchisquare, wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = (
    "curve-extrapolation-based-stopping-criteria-within-the-BO/"
    "results/final_results.csv"
)

OUTPUT_DIR = (
    "curve-extrapolation-based-stopping-criteria-within-the-BO/"
    "results/analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

METHOD_ORDER = [
    "no_early_stopping",
    "conventional_early_stopping",
    "extrapolation",
]

METHOD_LABELS = {
    "no_early_stopping": "No early stopping",
    "conventional_early_stopping": "Conventional early stopping",
    "extrapolation": "Curve extrapolation",
}

warnings.filterwarnings("ignore")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(RESULTS_PATH)

print("=" * 70)
print("RESULTS LOADED")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Datasets: {df['dataset'].unique()}")
print(f"Methods: {df['method'].unique()}")
print(f"Seeds: {df['seed'].nunique()}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CLEAN / STANDARDIZE
# ============================================================

df["method_label"] = df["method"].map(METHOD_LABELS)

df["method"] = pd.Categorical(
    df["method"],
    categories=METHOD_ORDER,
    ordered=True
)

df = df.sort_values(
    ["dataset", "seed", "method"]
)


# ============================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

summary = (
    df
    .groupby(["dataset", "method"], observed=True)
    .agg(
        n=("best_mae", "count"),

        mean_mae=("best_mae", "mean"),
        std_mae=("best_mae", "std"),
        median_mae=("best_mae", "median"),
        min_mae=("best_mae", "min"),
        max_mae=("best_mae", "max"),

        mean_time=("execution_time", "mean"),
        std_time=("execution_time", "std"),
        median_time=("execution_time", "median"),
        min_time=("execution_time", "min"),
        max_time=("execution_time", "max"),

        mean_best_trial=("best_trial", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"),
    index=False
)

print(summary.to_string(index=False))


# ============================================================
# 2. TIME AND MAE RELATIVE TO BASELINES
# ============================================================

relative_results = []

for dataset in df["dataset"].unique():

    dataset_df = df[df["dataset"] == dataset]

    baseline_no_es = dataset_df[
        dataset_df["method"] == "no_early_stopping"
    ].set_index("seed")

    baseline_es = dataset_df[
        dataset_df["method"] == "conventional_early_stopping"
    ].set_index("seed")

    extrapolation = dataset_df[
        dataset_df["method"] == "extrapolation"
    ].set_index("seed")

    for seed in extrapolation.index:

        if seed not in baseline_no_es.index:
            continue

        if seed not in baseline_es.index:
            continue

        ext_mae = extrapolation.loc[seed, "best_mae"]
        ext_time = extrapolation.loc[seed, "execution_time"]

        no_es_mae = baseline_no_es.loc[seed, "best_mae"]
        no_es_time = baseline_no_es.loc[seed, "execution_time"]

        es_mae = baseline_es.loc[seed, "best_mae"]
        es_time = baseline_es.loc[seed, "execution_time"]

        relative_results.append({
            "dataset": dataset,
            "seed": seed,

            "extrapolation_mae": ext_mae,
            "no_es_mae": no_es_mae,
            "conventional_es_mae": es_mae,

            "extrapolation_time": ext_time,
            "no_es_time": no_es_time,
            "conventional_es_time": es_time,

            "mae_diff_vs_no_es":
                ext_mae - no_es_mae,

            "mae_relative_vs_no_es_pct":
                100 * (ext_mae - no_es_mae) / no_es_mae,

            "mae_diff_vs_conventional_es":
                ext_mae - es_mae,

            "mae_relative_vs_conventional_es_pct":
                100 * (ext_mae - es_mae) / es_mae,

            "time_saved_vs_no_es_pct":
                100 * (no_es_time - ext_time) / no_es_time,

            "time_saved_vs_conventional_es_pct":
                100 * (es_time - ext_time) / es_time,
        })


relative_df = pd.DataFrame(relative_results)

relative_df.to_csv(
    os.path.join(OUTPUT_DIR, "relative_results.csv"),
    index=False
)

print("\n")
print("=" * 70)
print("RELATIVE RESULTS")
print("=" * 70)

print(
    relative_df
    .groupby("dataset")
    .agg({
        "mae_diff_vs_no_es": "mean",
        "mae_relative_vs_no_es_pct": "mean",
        "mae_diff_vs_conventional_es": "mean",
        "mae_relative_vs_conventional_es_pct": "mean",
        "time_saved_vs_no_es_pct": "mean",
        "time_saved_vs_conventional_es_pct": "mean",
    })
    .round(4)
)


# ============================================================
# 3. BOXPLOT: MAE
# ============================================================

for dataset in df["dataset"].unique():

    subset = df[df["dataset"] == dataset].copy()

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=subset,
        x="method_label",
        y="best_mae",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER]
    )

    sns.stripplot(
        data=subset,
        x="method_label",
        y="best_mae",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        color="black",
        alpha=0.5,
        size=4
    )

    plt.title(f"Validation MAE by method — {dataset}")
    plt.xlabel("Method")
    plt.ylabel("Best validation MAE")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_mae_boxplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 4. BOXPLOT: EXECUTION TIME
# ============================================================

for dataset in df["dataset"].unique():

    subset = df[df["dataset"] == dataset].copy()

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=subset,
        x="method_label",
        y="execution_time",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER]
    )

    sns.stripplot(
        data=subset,
        x="method_label",
        y="execution_time",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        color="black",
        alpha=0.5,
        size=4
    )

    plt.title(f"Execution time by method — {dataset}")
    plt.xlabel("Method")
    plt.ylabel("Execution time (s)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_time_boxplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. MEAN MAE COMPARISON
# ============================================================

for dataset in df["dataset"].unique():

    subset = (
        summary[summary["dataset"] == dataset]
        .copy()
    )

    subset["label"] = subset["method"].map(METHOD_LABELS)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=subset,
        x="label",
        y="mean_mae",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER]
    )

    plt.title(f"Mean validation MAE — {dataset}")
    plt.xlabel("Method")
    plt.ylabel("Mean best validation MAE")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_mean_mae.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 6. MEAN EXECUTION TIME COMPARISON
# ============================================================

for dataset in df["dataset"].unique():

    subset = (
        summary[summary["dataset"] == dataset]
        .copy()
    )

    subset["label"] = subset["method"].map(METHOD_LABELS)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=subset,
        x="label",
        y="mean_time",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER]
    )

    plt.title(f"Mean execution time — {dataset}")
    plt.xlabel("Method")
    plt.ylabel("Mean execution time (s)")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_mean_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. QUALITY VS COMPUTATIONAL COST
# ============================================================

for dataset in df["dataset"].unique():

    subset = summary[
        summary["dataset"] == dataset
    ].copy()

    plt.figure(figsize=(9, 6))

    for _, row in subset.iterrows():

        plt.scatter(
            row["mean_time"],
            row["mean_mae"],
            s=120,
            label=METHOD_LABELS[row["method"]]
        )

        plt.annotate(
            METHOD_LABELS[row["method"]],
            (
                row["mean_time"],
                row["mean_mae"]
            ),
            xytext=(8, 8),
            textcoords="offset points"
        )

    plt.xlabel("Mean execution time (s)")
    plt.ylabel("Mean best validation MAE")
    plt.title(
        f"Quality vs computational cost — {dataset}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_quality_vs_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. SEED-BY-SEED COMPARISON
# ============================================================

for dataset in df["dataset"].unique():

    subset = df[
        df["dataset"] == dataset
    ].copy()

    plt.figure(figsize=(12, 6))

    for method in METHOD_ORDER:

        method_df = subset[
            subset["method"] == method
        ]

        plt.plot(
            method_df["seed"],
            method_df["best_mae"],
            marker="o",
            label=METHOD_LABELS[method]
        )

    plt.xlabel("Seed")
    plt.ylabel("Best validation MAE")
    plt.title(
        f"Performance across random seeds — {dataset}"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{dataset}_seed_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 9. PAIRED STATISTICAL TESTS
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

statistical_results = []

for dataset in df["dataset"].unique():

    subset = df[
        df["dataset"] == dataset
    ].pivot(
        index="seed",
        columns="method",
        values="best_mae"
    )

    subset = subset.dropna()

    # --------------------------------------------------------
    # Friedman
    # --------------------------------------------------------

    if len(subset) >= 3:

        statistic, p_value = friedmanchisquare(
            subset["no_early_stopping"],
            subset["conventional_early_stopping"],
            subset["extrapolation"]
        )

        statistical_results.append({
            "dataset": dataset,
            "comparison": "Friedman",
            "method_a": "all",
            "method_b": "all",
            "statistic": statistic,
            "p_value": p_value
        })

        print(
            f"\n{dataset} — Friedman"
        )

        print(
            f"Statistic = {statistic:.4f}"
        )

        print(
            f"p-value = {p_value:.6f}"
        )

    # --------------------------------------------------------
    # Pairwise Wilcoxon
    # --------------------------------------------------------

    pairs = [
        (
            "extrapolation",
            "no_early_stopping"
        ),
        (
            "extrapolation",
            "conventional_early_stopping"
        ),
        (
            "conventional_early_stopping",
            "no_early_stopping"
        ),
    ]

    for method_a, method_b in pairs:

        if method_a not in subset.columns:
            continue

        if method_b not in subset.columns:
            continue

        statistic, p_value = wilcoxon(
            subset[method_a],
            subset[method_b]
        )

        statistical_results.append({
            "dataset": dataset,
            "comparison": "Wilcoxon",
            "method_a": method_a,
            "method_b": method_b,
            "statistic": statistic,
            "p_value": p_value
        })

        print(
            f"{method_a} vs {method_b}: "
            f"stat={statistic:.4f}, "
            f"p={p_value:.6f}"
        )


statistics_df = pd.DataFrame(statistical_results)

statistics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "statistical_tests.csv"
    ),
    index=False
)


# ============================================================
# 10. TIME SAVINGS SUMMARY
# ============================================================

time_summary = (
    relative_df
    .groupby("dataset")
    .agg(
        mean_time_saved_vs_no_es=(
            "time_saved_vs_no_es_pct",
            "mean"
        ),

        std_time_saved_vs_no_es=(
            "time_saved_vs_no_es_pct",
            "std"
        ),

        mean_time_saved_vs_es=(
            "time_saved_vs_conventional_es_pct",
            "mean"
        ),

        std_time_saved_vs_es=(
            "time_saved_vs_conventional_es_pct",
            "std"
        ),
    )
    .reset_index()
)

time_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "time_savings_summary.csv"
    ),
    index=False
)


# ============================================================
# 11. LATEX TABLE — DESCRIPTIVE RESULTS
# ============================================================

latex_summary = summary.copy()

latex_summary["method"] = (
    latex_summary["method"]
    .map(METHOD_LABELS)
)

latex_table = latex_summary[
    [
        "dataset",
        "method",
        "mean_mae",
        "std_mae",
        "mean_time",
        "std_time"
    ]
].copy()

latex_table.columns = [
    "Dataset",
    "Method",
    "Mean MAE",
    "Std. MAE",
    "Mean time (s)",
    "Std. time (s)"
]

latex_code = latex_table.to_latex(
    index=False,
    float_format="%.4f"
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_results.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(latex_code)


# ============================================================
# 12. LATEX TABLE — TIME SAVINGS
# ============================================================

latex_time = time_summary.copy()

latex_time.columns = [
    "Dataset",
    "Time saved vs. no ES (\%)",
    "Std.",
    "Time saved vs. conventional ES (\%)",
    "Std."
]

latex_time_code = latex_time.to_latex(
    index=False,
    float_format="%.2f"
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "time_savings.tex"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(latex_time_code)


# ============================================================
# 13. AUTOMATIC TEXT REPORT
# ============================================================

report_lines = []

report_lines.append(
    "AUTOMATIC EXPERIMENT ANALYSIS REPORT"
)

report_lines.append(
    "=" * 70
)

for dataset in df["dataset"].unique():

    report_lines.append("")
    report_lines.append(
        f"DATASET: {dataset}"
    )

    dataset_summary = summary[
        summary["dataset"] == dataset
    ]

    for _, row in dataset_summary.iterrows():

        method = METHOD_LABELS[row["method"]]

        report_lines.append(
            f"- {method}: "
            f"MAE={row['mean_mae']:.6f} "
            f"(std={row['std_mae']:.6f}), "
            f"time={row['mean_time']:.2f}s "
            f"(std={row['std_time']:.2f}s)"
        )

    rel = relative_df[
        relative_df["dataset"] == dataset
    ]

    if len(rel) > 0:

        report_lines.append(
            f"- Extrapolation time saving vs no ES: "
            f"{rel['time_saved_vs_no_es_pct'].mean():.2f}%"
        )

        report_lines.append(
            f"- Extrapolation time saving vs conventional ES: "
            f"{rel['time_saved_vs_conventional_es_pct'].mean():.2f}%"
        )

        report_lines.append(
            f"- Extrapolation MAE difference vs no ES: "
            f"{rel['mae_diff_vs_no_es'].mean():.6f}"
        )

        report_lines.append(
            f"- Extrapolation MAE difference vs conventional ES: "
            f"{rel['mae_diff_vs_conventional_es'].mean():.6f}"
        )


with open(
    os.path.join(
        OUTPUT_DIR,
        "automatic_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report_lines))


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Results saved to:\n{OUTPUT_DIR}"
)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {filename}")