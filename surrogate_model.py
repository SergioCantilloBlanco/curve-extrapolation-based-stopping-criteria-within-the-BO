
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class SugorrateModel:

    def __init__(self):
        #kernel = RBF(1.0)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.model = GaussianProcessRegressor(kernel=kernel,
            alpha=1e-6,)

    def fit(self, X, y):
        X = np.array(X).reshape(-1,1)
        self.model.fit(X, y)

    def predict(self, X):
        X = np.array(X).reshape(-1,1)
        return self.model.predict(X, return_std=True)