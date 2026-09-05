import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from curves import double_exponential
from extrapolationPruner import extrapolationPruner
from xgbCustomCallback import EarlyStopWithPrediction, customCallback
import numpy as np


SEED = [1,2,3,4,5,10,20,40,60,100,200,400,600,800,1000]
N_TRIALS = 35
TEST_SIZE = 0.2


def train_xgboost(num_boost_round, trial, study, params, dtrain, dval):
    evals_result = {}

    model = xgb.train(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        evals_result=evals_result,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[customCallback(trial, study)],
        verbose_eval=False,
    )

    return evals_result

results = []
early_stopped_trials = []
df = pd.read_csv(
        "curve-extrapolation-based-stopping-criteria-within-the-BO/data/winequality-red.csv"
    )
for SEED in SEED:
    np.random.seed(SEED)
    y = df.quality
    X = df.iloc[:, :-1]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)


    def objective_hpo(trial):
        num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)

        param = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "seed": SEED,
        }
        try:
            evals_result = train_xgboost(num_boost_round, trial, study, param, dtrain, dval)
            trial.set_user_attr("early_stopped", False)
            return min(evals_result["val"]["mae"])
        except EarlyStopWithPrediction as e:
            early_stopped_trials.append({
            "seed": SEED,
            "trial": trial.number,
            "predicted_best": e.predicted_value,
            "rounds_seen": len(e.val_mae_history),
            "num_boost_round": trial.params["num_boost_round"],
            "last_observed_mae": e.val_mae_history[-1],
            "best_so_far": e.best_so_far,
            "params": trial.params,
            })
            trial.set_user_attr("early_stopped", True)
            return e.predicted_value


    sampler = optuna.samplers.GPSampler(seed=SEED)
    study = optuna.create_study(
        sampler=sampler,
        direction="minimize",
        pruner=extrapolationPruner(
            curve_model=double_exponential,
        ),
    )

    start_time = time.time()
    study.optimize(
        objective_hpo,
        n_trials=N_TRIALS,
    )
    finish_time = time.time()

    n_early_stopped = sum(
        1 for t in study.trials if t.user_attrs.get("early_stopped")
    )

    results.append({
            "seed": SEED,
            "best_mae": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "trials": len(study.trials),
            "early_stopped_trials": n_early_stopped,
            "execution_time": finish_time - start_time
        })

results_df = pd.DataFrame(results)
early = pd.DataFrame(early_stopped_trials)
results_df.to_csv('curve-extrapolation-based-stopping-criteria-within-the-BO/results/extrapolation_results.csv', index=False)
early.to_csv('curve-extrapolation-based-stopping-criteria-within-the-BO/results/early_stopped_trials.csv', index=False)
