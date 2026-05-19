
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sklearn
from acquisition_function import acquisition_lcb
from surrogate_model import SurrogateModel


class BayesianOptimizer:

    def __init__(self,objective,bounds,num_samples,num_iter, surrogate, acquisition, stopping = None ) -> None:
        self.objective = objective
        self.bounds = bounds
        self.num_iter = num_iter
        self.num_samples = num_samples
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.stopping = stopping
        self.best_value_y = []
        self.X = []
        self.y = []

    def initialize(self):
        for _ in range(self.num_samples):
            x = np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            #x = np.random.uniform(self.bounds[0], self.bounds[-1])
            y = self.objective(x)

            self.X.append(x)
            self.y.append(y)
    
    def oned_plot(self, y_pred, y_std,new_x,new_y,i):
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10, 5))
        ax1.plot(self.bounds, self.objective(self.bounds), label='Black Box Function')
        ax1.scatter(self.X, self.y, color='red', label='Samples')
        ax1.scatter(new_x, new_y, color='blue', label='New point')
        ax1.plot(self.bounds, y_pred, color='blue', label='Gaussian Process')
        ax1.fill_between(self.bounds, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
        ax1.set(title=f'Black Box Function with Gaussian Process Surrogate Model Loop Nº{i}', ylabel='Black Box Output', xlabel='x')
        ax2.plot(np.arange(len(self.best_value_y)), self.best_value_y,color='orange', label='convergence curve')
        ax2.scatter(np.arange(len(self.best_value_y)), self.best_value_y, color='blue', label='samples')

    def multid_plot(self, y_pred, y_std,new_x,new_y,i):
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10, 5))

        x_plot = np.linspace(self.bounds[0,0], self.bounds[0,1], 200)
        x_test = []

        for x in x_plot:
                point = np.mean(self.bounds, axis=1)
                point[0] = x
                x_test.append(point)

        x_test = np.array(x_test)

        y_true = np.array([self.objective(x) for x in x_test])
        ax1.plot(x_plot, y_true, label="True function slice")

        y_pred, y_std = self.surrogate.predict(x_test)
        ax1.plot(x_plot, y_pred, color='blue', label='GP Mean')

        ax1.fill_between(x_plot,y_pred - 2*y_std,y_pred + 2*y_std,alpha=0.2)

        X = np.array(self.X)
        ax1.scatter(X[:, 0], self.y, color='red', label='Samples')
        ax1.scatter([new_x[0]], [new_y], color='blue', label='New point')
        ax1.set_title(f'Iteration {i}')
        ax1.legend()

        ax2.plot(self.best_value_y, color='orange')
        ax2.set_title("Convergence")

        plt.show()

    def loop(self):
        for i in range(self.num_iter):
            X = np.array(self.X)
            y = np.array(self.y)
            self.surrogate.fit(X, y)

            candidates = np.random.uniform([b[0] for b in self.bounds],[b[1] for b in self.bounds],size=(1000, len(self.bounds)))
            y_pred, y_std = self.surrogate.predict(candidates)
            ucb = self.acquisition(y_pred, y_std)

            new_x = candidates[np.argmin(ucb)]
            new_y = self.objective(new_x)

            self.X.append(new_x)
            self.y.append(new_y)
            self.best_value_y.append(np.min(self.y))

            self.multid_plot(y_pred,y_std,new_x,new_y,i)



def black_box_function(x) -> float:
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

if __name__ == "__main__":
    #x_range = np.linspace(-2*np.pi, 2*np.pi, 200)
    x_bounds = np.array([(-15,20)])
    surrogate_model = SurrogateModel()
    optimizer = BayesianOptimizer(black_box_function_2,x_bounds,5,30,surrogate_model,acquisition_lcb)
    optimizer.initialize()
    optimizer.loop()






