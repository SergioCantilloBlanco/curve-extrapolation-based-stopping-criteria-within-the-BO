
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


class SugorrateModel:

    def __init__(self):
        kernel = RBF(1.0)
        self.model = GaussianProcessRegressor(kernel=kernel)

    def fit(self, X, y):
        X = np.array(X).reshape(-1,1)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X, return_std=True)