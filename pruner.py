import numpy as np
from optuna.pruners import BasePruner
from scipy.optimize import curve_fit


def exp_decay(x, a, b, c):
    '''
    a -> Controls vertical distance between start and asymptote
    b -> Controls how fast decay happens
    c -> Value the curve approaches as iterations go to infinity
    '''
    return a * np.exp(-b * x) + c

class ExtrapolationPruner(BasePruner):
    def __init__(self, epsilon, min_steps):
        self.epsilon = epsilon
        self.min_steps = min_steps

    def prune(self, study, trial) -> bool:
        values_dict = trial.intermediate_values
        steps = sorted(values_dict.keys())
        y_data = [values_dict[s] for s in steps]

        if self.min_steps <= len(steps):
            step = trial.last_step
            if step:
                
                this_score = trial.intermediate_values[step]

                a = y_data[0] - study.best_value
                b = 0.1
                c = study.best_value

                x_data = np.arange(len(steps))

                optimal_values, covariance = curve_fit(exp_decay, x_data, y_data, p0=[a,b,c], maxfev=10000)

                dif = abs(this_score - optimal_values[-1])

                if self.epsilon < dif:
                    print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                    return True


        return False