import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import optuna
import pandas as pd
import xgboost as xgb
import numpy as np

from extrapolationPruner import extrapolationPruner
from xgbCustomCallback import EarlyStopWithPrediction, customCallback

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [1, 2, 3, 4, 5, 10, 20, 40, 60, 100, 200, 400, 600, 800, 1000]

N_TRIALS = 35
TEST_SIZE = 0.2
CONVENTIONAL_EARLY_STOPPING = 30

RESULTS_PATH = (
    "curve-extrapolation-based-stopping-criteria-within-the-BO/"
    "results/final_results.csv"
)

# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "stroke_dataset": {
    "path": (
        "curve-extrapolation-based-stopping-criteria-within-the-BO/"
        "data/healthcare-dataset-stroke-data.csv"
    ),
    "target": "stroke",
    "task": "binary:logistic",
    "target_mapping": None,
    "eval_metric": "logloss",
    },
    "loan_data": {
    "path": (
        "curve-extrapolation-based-stopping-criteria-within-the-BO/"
        "data/loan_data.csv"
    ),
    "target": "loan_status",
    "task": "binary:logistic",
    "target_mapping": None,
    "eval_metric": "logloss",
    },


    "wine_quality_red": {
        "path": (
            "curve-extrapolation-based-stopping-criteria-within-the-BO/"
            "data/winequality-red.csv"
        ),
        "target": "quality",
        "task": "reg:squarederror",
        "target_mapping": None,
        "eval_metric": "mae",
    },
}


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(dataset_config):
    df = pd.read_csv(dataset_config["path"])
    target = dataset_config["target"]

    X = df.drop(columns=[target])
    y = df[target]

    target_mapping = dataset_config.get("target_mapping")

    if target_mapping is not None:
        y = y.map(target_mapping)

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns

    if len(categorical_columns) > 0:
        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            dtype=int
        )

    return X, y


# Xgboost training function

def train_xgboost(num_boost_round,
    params,
    dtrain,
    dval,
    early_stopping_rounds=None,
    callbacks=None):

    evals_result = {}

    model = xgb.train(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        evals_result=evals_result,
        evals=[
            (dtrain, "train"),
            (dval, "val")
        ],
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
        verbose_eval=False
    )

    return model, evals_result

#Hyperparameters suggestion function

def suggest_hyperparameters(trial, seed, config):   
    param = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "objective": config["task"],
                "eval_metric":config["eval_metric"],
                "seed": seed,
            }

    num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)
    return param, num_boost_round

# ============================================================
# BASELINE 1
# NO EARLY STOPPING
# ============================================================

def run_no_early_stopping(
    trial,
    study,
    seed,
    dtrain,
    dval,
    dataset_config
):

    params, num_boost_round = suggest_hyperparameters(
        trial,
        seed,
        dataset_config
    )

    model, evals_result = train_xgboost(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        dval=dval,
        early_stopping_rounds=None
    )
    metric = dataset_config["eval_metric"]

    best_mae = min(
        evals_result["val"][metric]
    )

    return best_mae

# ============================================================
# BASELINE 2
# CONVENTIONAL EARLY STOPPING
# ============================================================

def run_conventional_early_stopping(
    trial,
    study,
    seed,
    dtrain,
    dval,
    dataset_config
):

    params, num_boost_round = suggest_hyperparameters(
        trial,
        seed,
        dataset_config
    )

    model, evals_result = train_xgboost(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        dval=dval,
        early_stopping_rounds=CONVENTIONAL_EARLY_STOPPING
    )
    metric = dataset_config["eval_metric"]

    best_mae = min(
        evals_result["val"][metric]
    )

    return best_mae

# ============================================================
# METHOD 3
# CURVE EXTRAPOLATION
# ============================================================

def run_extrapolation(
    trial,
    study,
    seed,
    dtrain,
    dval,
    dataset_config
):

    params, num_boost_round = suggest_hyperparameters(
        trial,
        seed,
        dataset_config
    )
    try:
        #previous_best = study.best_value if len(study.trials) > 0 else np.nan
        metric = dataset_config["eval_metric"]
        model, evals_result = train_xgboost(num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        dval=dval,
        early_stopping_rounds=None,
        callbacks=[customCallback(trial, study, metric)]
        )

        return min(evals_result["val"][metric])
    
    except EarlyStopWithPrediction as e:
            return e.predicted_value




# ============================================================
# GENERIC EXPERIMENT RUNNER
# ============================================================

METHODS = {
    "no_early_stopping": 
        {"func":run_no_early_stopping,
         "pruner": None},
    "conventional_early_stopping":
        {"func":run_conventional_early_stopping,
         "pruner": None},
    "extrapolation":
        {"func":run_extrapolation,
         "pruner": extrapolationPruner()}
}

# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    dataset_name,
    X,
    y,
    seed,
    method
):
    print(
        f"Dataset={dataset_name} | "
        f"Method={method} | "
        f"Seed={seed}"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=1 - TEST_SIZE,
        test_size=TEST_SIZE,
        random_state=seed
    )

    dtrain = xgb.DMatrix(X_train,label=y_train)
    dval = xgb.DMatrix(X_val,label=y_val)

    #Objective

    method_function = METHODS[method]["func"]

    def objective(trial):

        return method_function(
            trial=trial,
            study=study,
            seed=seed,
            dtrain=dtrain,
            dval=dval,
            dataset_config=DATASETS[dataset_name]
        )

    sampler = optuna.samplers.GPSampler(
        seed=seed
    )
    study = optuna.create_study(
        sampler=sampler,
        direction="minimize",
        pruner = METHODS[method]["pruner"]

    )
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=N_TRIALS
    )
    finish_time = time.time()

    best_trial = study.best_trial

    result = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "best_mae": best_trial.value,
        "best_trial": best_trial.number,
        "trials": len(study.trials),
        "execution_time":
            finish_time - start_time,
        "best_params":
            best_trial.params,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    results = []

    total_experiments = (
        len(DATASETS)
        * len(METHODS)
        * len(SEEDS)
    )

    experiment_number = 0

    # --------------------------------------------------------
    # Dataset loop
    # --------------------------------------------------------

    for dataset_name, dataset_config in DATASETS.items():

        print("\n")
        print("=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        X, y = load_dataset(
            dataset_config
        )

        for method in METHODS:

            print("\n")
            print(
                f"METHOD: {method}"
            )

            # ------------------------------------------------
            # Seed loop
            # ------------------------------------------------

            for seed in SEEDS:

                experiment_number += 1

                print(
                    f"\n[{experiment_number}/"
                    f"{total_experiments}]"
                )

                result = run_experiment(
                    dataset_name=dataset_name,
                    X=X,
                    y=y,
                    seed=seed,
                    method=method,
                )

                results.append(result)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("EXPERIMENTS COMPLETED")
    print("=" * 70)

    print(
        f"Results saved to:\n{RESULTS_PATH}"
    )

    print(
        f"Total rows: {len(results_df)}"
    )


if __name__ == "__main__":
    main()