import numpy as np
from scipy.stats import norm


def acquisition_lcb(predicted_mean, predicted_std_deviation,current_best= None, kappa=5.0):
    return predicted_mean - kappa*predicted_std_deviation

def expected_improvement(predicted_mean, predicted_std_deviation, current_best, kappa=None):
    if np.any(predicted_std_deviation == 0):
        predicted_std_deviation = np.maximum(predicted_std_deviation, 1e-9)

    improvement = current_best - predicted_mean

    z = improvement / predicted_std_deviation
    ei = improvement * norm.cdf(z) + predicted_std_deviation * norm.pdf(z)

    return np.where(predicted_std_deviation > 1e-9, ei, np.maximum(0, improvement))

