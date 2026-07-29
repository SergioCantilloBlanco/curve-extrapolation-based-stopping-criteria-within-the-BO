
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class SurrogateModel:

    def __init__(self):
        #kernel = RBF(1.0)
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            RBF(length_scale=5.0, length_scale_bounds=(0.5, 50.0))
        )
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=5, normalize_y=True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X, return_std=True)