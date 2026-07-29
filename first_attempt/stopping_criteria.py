import numpy as np


def exp_decay(x, a, b, c):
    '''
    a -> Controls vertical distance between start and asymptote
    b -> Controls how fast decay happens
    c -> Value the curve approaches as iterations go to infinity
    '''
    return a * np.exp(-b * x) + c

class PercentageGainStopping:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def should_stop(self, optimal_values, recorded_points, covariance):
        percentage_gain = ((recorded_points.best_y[-1] - optimal_values[-1]) / recorded_points.best_y[-1])*100
        absolute_gain = abs(recorded_points.best_y[-1] - optimal_values[-1])
        c_std = np.sqrt(covariance[2,2])
        print(f"Relative percentage gain:{percentage_gain}")
        print(f"Absolute gain:{absolute_gain}")
        if absolute_gain < self.epsilon:
            print(f"Best value: {recorded_points.best_y[-1]}")
            return True
        return False
