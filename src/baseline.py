import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna.visualization as vis
import time

SEED = [1,2,3,4,5,10,20,40,60,100,200,400,600,800,1000]
N_TRIALS = 35
TEST_SIZE = 0.2


def train_xgboost(num_boost_round, params, dtrain, dval):
    evals_result = {}
    model = xgb.train(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        evals_result=evals_result,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=25,
        verbose_eval=False
    )
    return evals_result
results = []

df = pd.read_csv('curve-extrapolation-based-stopping-criteria-within-the-BO/data/winequality-red.csv')
for SEED in SEED:
    y = df.quality
    X = df.iloc[:,:-1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=TEST_SIZE,
                                                        random_state=SEED)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def objective_hpo(trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "seed": SEED,
        }

        num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)

        evals_result = train_xgboost(num_boost_round, param, dtrain, dval)


        return min(evals_result["val"]["mae"])


    sampler = optuna.samplers.GPSampler(seed=SEED)
    study = optuna.create_study(sampler= sampler, direction='minimize')
    start_time = time.time()
    study.optimize(objective_hpo, n_trials=N_TRIALS)
    finish_time = time.time()

    # get the best trial
    trial = study.best_trial
    results.append({
        "seed": SEED,
        "best_mae": trial.value,
        "best_params": trial.params,
        "best_trial": trial.number,
        "trials": len(study.trials),
        "execution_time": finish_time-start_time
    })
results_df = pd.DataFrame(results)
results_df.to_csv('curve-extrapolation-based-stopping-criteria-within-the-BO/results/baseline_results.csv', index=False)

