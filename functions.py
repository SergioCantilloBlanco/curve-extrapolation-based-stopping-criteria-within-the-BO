import numpy as np
import math

def reverse_sigmoid(x, L, k, x0, b):
    '''
    - x0 para el desplazamiento horizontal
    k * factor de inclinacion
    (L - b) estira la curva desde el punto mas alto (L) hasta el mas bajo (b)
    b + levanta la curva
    '''
    if math.isnan(x0):
        x0 = 0
    z = b + (L - b) / (1 + np.exp(k * (x - x0)))
    return z

def exp_decay(x, a, b, c):
    '''
    a -> Controls vertical distance between start and asymptote
    b -> Controls how fast decay happens
    c -> Value the curve approaches as iterations go to infinity
    '''
    return a * np.exp(-b * x) + c

def power_law(x, a, b, c, d):
    return a / np.power(x + b, c) + d

def black_box_function_1d(x) -> float:
    y = np.sin(x) + np.cos(2*x)
    return y

def black_box_function_2(x) -> float:
    if isinstance(x, float):
        x = [x]

    w0 = 1 + (x[0] - 1) / 4
    term1 = np.power(np.sin(np.pi * w0), 2)

    term2 = 0
    for i in range(len(x) - 1):
        wi = 1 + (x[i] - 1) / 4
        term2 += np.power(wi - 1, 2) * (1 + 10 * np.power(np.sin(wi * np.pi + 1), 2))

    wd = 1 + (x[-1] - 1) / 4
    term3 = np.power(wd - 1, 2)
    term3 *= 1 + np.power(np.sin(2 * np.pi * wd), 2)

    y = term1 + term2 + term3
    return y