import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from curves import double_exponential
import optuna
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

SEED = 100
N_TRIALS = 50
TEST_SIZE = 0.2
observable_percentage = 0.4

def train_xgboost(num_boost_round, params, dtrain, dval):
    evals_result = {}
    model = xgb.train(
        num_boost_round=num_boost_round,
        params=params,
        dtrain=dtrain,
        evals_result=evals_result,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=False
    )
    return model, evals_result

def fit_curve(OBSERVED_ROUNDS, val_mae_history, trial_number):
    val_mae_history = val_mae_history[::2]
    lists = enumerate(val_mae_history)
    x, y = zip(*lists)
    results = []

    extrapolation_x = x[:OBSERVED_ROUNDS]
    extrapolation_y = y[:OBSERVED_ROUNDS]

    future_x = x[OBSERVED_ROUNDS:]
    future_y = y[OBSERVED_ROUNDS:]

    y0 = extrapolation_y[0]
    yend = extrapolation_y[-1]
    amp = y0 - yend

    lower_bounds = [0,0,0,0,0]
    upper_bounds = [np.inf,5,np.inf,5,extrapolation_y[-1]]

    curve = {"Double Exp": (double_exponential,[amp*0.7,0.5,amp*0.3,0.03,yend]),}

    for _, (model_name, (model_func, initial_params)) in enumerate(curve.items()):
        try:
            optimal_values, covariance = curve_fit(
                model_func,
                extrapolation_x,
                extrapolation_y,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                maxfev=1000
            )

            predicted_y = model_func(np.array(future_x), *optimal_values)
            results.append({
                "trial": trial_number,
                "success": True,
                "predicted_final": predicted_y[-1],
                "actual_final": future_y[-1],
                "error": abs(predicted_y[-1] - future_y[-1]),
            })
        except Exception as e:
            results.append({
                "trial": trial_number,
                "success": False,
                "predicted_final": np.nan,
                "actual_final": np.nan,
                "error": np.nan,
            })
            print(f"Error fitting {model_name}: {e}")
            continue

    return results, optimal_values[-1]

df = pd.read_csv('curve-extrapolation-based-stopping-criteria-within-the-BO/data/winequality-red.csv')
y = df.quality
X = df.iloc[:,:-1]
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=TEST_SIZE,
                                                      random_state=SEED)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

all_evals_results = []

def objective_hpo(trial):
    num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)
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


    model, evals_result = train_xgboost(num_boost_round, param, dtrain, dval)
    all_evals_results.append(evals_result['val']['mae'])

    return evals_result["val"]["mae"][-1]


sampler = optuna.samplers.GPSampler(seed=SEED)
study = optuna.create_study(sampler= sampler, direction='minimize')
study.optimize(objective_hpo, n_trials=N_TRIALS)

best_so_far = np.inf
predicted_prune_trials = []
true_prune_trials = []
pruning_results = []
for trial_number, eval in enumerate(all_evals_results):
    results, c = fit_curve(int(np.floor(len(eval)*observable_percentage)), eval, trial_number)
    if results[0]["predicted_final"] is not None:
        pruning_results.append({"trial": trial_number,
                                "predicted_final": results[0]["predicted_final"],
                                "asymptote": c,
                                "actual_final": results[0]["actual_final"],
                                "error": results[0]["error"],
                                "predicted_prune": results[0]["predicted_final"] > best_so_far + 0.006,
                                "true_prune": results[0]["actual_final"] > best_so_far})
    best_so_far = min(best_so_far, eval[-1])

true_prune_trials = [result["true_prune"] for result in pruning_results]
predicted_prune_trials = [result["predicted_prune"] for result in pruning_results]


cm = confusion_matrix(true_prune_trials, predicted_prune_trials)
print([result["predicted_final"] for result in pruning_results])
print([result["actual_final"] for result in pruning_results])
print("Accuracy :", accuracy_score(true_prune_trials, predicted_prune_trials))
print("Precision:", precision_score(true_prune_trials, predicted_prune_trials))
print("Recall   :", recall_score(true_prune_trials, predicted_prune_trials))
print("F1       :", f1_score(true_prune_trials, predicted_prune_trials))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Pruned", "Pruned"])
disp.plot(cmap=plt.cm.Blues)
plt.show()


                



    

