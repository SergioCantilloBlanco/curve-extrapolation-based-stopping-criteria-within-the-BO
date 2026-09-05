import numpy as np
import optuna
from optuna.pruners import BasePruner
from scipy.optimize import curve_fit



class extrapolationPruner(BasePruner):
    def __init__(
        self,
        curve_model,
        observation_fraction=[0.2],
    ):
        self.curve_model = curve_model
        self.observation_fraction = observation_fraction

    def fit_curve(self, val_mae_history):
        lists = enumerate(val_mae_history)
        x, y = zip(*lists)

        y0 = y[0]
        yend = y[-1]
        amp = y0 - yend

        lower_bounds = [0, 0, 0, 0, 0]
        upper_bounds = [np.inf, 5, np.inf, 5, y[-1]]

        try:
            optimal_values, covariance = curve_fit(
                self.curve_model,
                x,
                y,
                p0=[amp * 0.7, 0.5, amp * 0.3, 0.03, yend],
                bounds=(lower_bounds, upper_bounds),
                maxfev=1000,
            )
        except Exception as e:
            print(f"Error fitting curve: {e}")
            return None, None

        return optimal_values, covariance

    def prune(self, study, trial, intermediate_values):
        val_mae_history = intermediate_values
        total_rounds = trial.params["num_boost_round"]

        for fraction in self.observation_fraction:
            if len(val_mae_history) == int(total_rounds * fraction):
                break
        else:
            return False, None, None, None

        future_rounds = np.arange(
            len(val_mae_history),
            total_rounds,
        )

        result = self.fit_curve(val_mae_history[::2])

        if result is None:
            return False, None, None, None

        optimal_values, covariance = result

        prediction = self.curve_model(
            future_rounds, 
            *optimal_values,
        )
        predicted_best = min(
            np.concatenate([
                val_mae_history,
                prediction
            ])
        )

        completed = [
            t.value
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed:
            return False, None, None, None

        best_so_far = study.best_value
        
        sigma = np.sqrt(covariance[-1, -1])
        margin = 2 * sigma

        # prediction = optimal_values[-1]

        if predicted_best > best_so_far:
        #     print(
        #     f"trial={trial.number}",
        #     f"last={val_mae_history[-1]:.4f}",
        #     f"pred_final={self.curve_model(total_rounds-1,*optimal_values):.4f}",
        #     f"asymptote={optimal_values[-1]:.4f}",
        #     f"params={trial.params}"
        # )
            return True, predicted_best, val_mae_history, best_so_far

        return False, None, None, None