import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from curves import *
import optuna
import time

SEED = 100
N_TRIALS = 50
TEST_SIZE = 0.2
observable_percentage = 0.2

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

def fit_curve(OBSERVED_ROUNDS, val_mae_history, trial_number, observed_percentage):
    lists = enumerate(val_mae_history)
    x, y = zip(*lists)
    results = []

    extrapolation_x = x[:OBSERVED_ROUNDS]
    extrapolation_y = y[:OBSERVED_ROUNDS]

    future_x = x[OBSERVED_ROUNDS:]
    future_y = y[OBSERVED_ROUNDS:]
    remaining_rounds = len(future_x)

    y0 = extrapolation_y[0]
    yend = extrapolation_y[-1]
    amp = y0 - yend

    # lower_bounds = [0,0,0,0,0]
    # upper_bounds = [np.inf,5,np.inf,5,yend]

    models = {
        "Double Exp": (double_exponential,[amp*0.7,0.5,amp*0.3,0.03,yend]),
        # "Exponential": (exponential,[amp,0.1,yend]),
        # "Power Law": (power_law,[amp*10,1.0,1.0,yend]),
        # "Logarithmic": (logarithmic,[y0,amp/np.log(OBSERVED_ROUNDS+1),1.0 ]),
        # "Rational": (rational,[amp*5,1.0,yend]),
        # "General Rational": (rational_general,[1.0, y0, 1.0, 1.0]),
        # "Weibull": (weibull,[amp, 0.1, 1.0, yend]),
        # "Stretched Exp": (stretched_exponential,[amp, 0.1, 0.8, yend]),
        # "Gompertz": (gompertz,[amp,-0.1,OBSERVED_ROUNDS/2,yend]),
        # "Logistic": (logistic,[amp,0.1,OBSERVED_ROUNDS/2,yend])
        }

    for _, (model_name, (model_func, initial_params)) in enumerate(models.items()):
        try:
            optimal_values, covariance = curve_fit(
                model_func,
                extrapolation_x,
                extrapolation_y,
                p0=initial_params,
                # bounds=(lower_bounds, upper_bounds),
                maxfev=1000
            )

            predicted_y = model_func(np.array(future_x), *optimal_values)

            results.append({
                "trial": trial_number,
                "curve": model_name,
                "observed_percentage": observed_percentage,
                "success": True,
                "predicted_final": predicted_y[-1],
                "actual_final": future_y[-1],
                "error": abs(predicted_y[-1] - future_y[-1]),
                "remaining_rounds": remaining_rounds
            })
        except Exception as e:
            results.append({
                "trial": trial_number,
                "curve": model_name,
                "observed_percentage": observed_percentage,
                "success": False,
                "predicted_final": np.nan,
                "actual_final": np.nan,
                "error": np.nan,
                "remaining_rounds": remaining_rounds
            })
            print(f"Error fitting {model_name}: {e}")
            continue

    return results



df = pd.read_csv('curve-extrapolation-based-stopping-criteria-within-the-BO/data/winequality-red.csv')
y = df.quality
X = df.iloc[:,:-1]
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=TEST_SIZE,
                                                      random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

all_results = []

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
    results = fit_curve(int(num_boost_round*observable_percentage), evals_result['val']['mae'], trial.number, observable_percentage)
    all_results.extend(results)



    return evals_result["val"]["mae"][-1]

sampler = optuna.samplers.GPSampler(seed=SEED)
study = optuna.create_study(sampler= sampler, direction='minimize')
start_time = time.time()
study.optimize(objective_hpo, n_trials=N_TRIALS)
finish_time = time.time()

results_df = pd.DataFrame(all_results)

summary = (
    results_df
    .groupby("curve")
    .agg(
        mean_error=("error", "mean"),
        std_error=("error", "std"),
        success=("success", "mean")
    )
)
print(summary)


# print(f"Observed Rounds: {OBSERVED_ROUNDS}")
# print(f"Actual final mae: {future_y[-1]}")
# print(f"Predicted final mae: {predicted_y[-1]}")
# print(f"Difference: {best_result}")
# print(f"Optimal Parameters: {optimal_values}")


fig, axs = plt.subplots(1, 2, figsize=(12,5))

axs[0].plot(x, y, label='Actual MAE')
axs[0].axvline(OBSERVED_ROUNDS-1, color='gray', linestyle='--',
        label='Observation cutoff')
axs[0].legend()
axs[0].set_xlabel('Boosting Round')
axs[0].set_ylabel('MAE')
axs[0].set_title('Validation MAE History')

# Extrapolation comparison
axs[1].plot(extrapolation_x, extrapolation_y, 'r.-', label='Observed')
axs[1].plot(future_x, future_y, 'g.-', label='Actual Future')
axs[1].plot(future_x, predicted_y, 'b--', linewidth=2, label='Predicted Future')

axs[1].set_xlabel('Boosting Round')
axs[1].set_ylabel('MAE')
axs[1].set_title('Curve Extrapolation')
axs[1].legend()


plt.tight_layout()
plt.show()