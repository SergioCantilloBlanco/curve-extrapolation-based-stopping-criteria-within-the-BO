import optuna
import optuna.visualization as vis
import pandas as pd
from plotly.io import show
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

from pruner import ExtrapolationPruner

df = pd.read_csv('data\winequality-red.csv')
y = df.quality
X = df.iloc[:,:-1]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


def objective_hpo(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    clf = XGBRegressor(**param)

    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose = False)

    predictions = clf.predict(X_valid)
    score = mean_absolute_error(predictions, y_valid)

    trial.report(score, trial.number)

    if trial.should_prune():
        raise optuna.TrialPruned()
        
    return score

sampler = optuna.samplers.GPSampler()
pruner = ExtrapolationPruner(epsilon = 0.5, min_steps = 5)
study = optuna.create_study(sampler= sampler, direction='minimize', pruner = pruner)
study.optimize(objective_hpo, n_trials=60)


fig = vis.plot_optimization_history(study)
fig.show()

# get the best trial
trial = study.best_trial
print(trial)

