import numpy as np


def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def double_exponential(x, a1, b1, a2, b2, c):
    return c + a1*np.exp(-b1*x) + a2*np.exp(-b2*x)

def power_law(x, a, b, p, c):
    return c + a / np.power(x + b, p)

def logarithmic(x, a, b, c):
    return a - b*np.log(x + c)

def rational(x, a, b, c):
    return c + a/(x + b)

def rational_general(x, a, b, c, d):
    return (a*x + b)/(c*x + d)

def weibull(x, a, b, k, c):
    return c + a*np.exp(-(b*x)**k)

def stretched_exponential(x, a, b, p, c):
    return c + a*np.exp(-(b*x)**p)

def gompertz(x, a, b, d, c):
    return c + a*np.exp(-np.exp(b*(x-d)))

def logistic(x, a, b, d, c):
    return c + a/(1 + np.exp(b*(x-d)))