import numpy as np
from scipy.stats import norm


def acquisition_ucb(predected_mean, predected_std_deviation, kappa=2.0):
    return predected_mean + kappa*predected_std_deviation

def acquisition_ei(predected_mean, predected_std_deviation, best_y):
    pass
