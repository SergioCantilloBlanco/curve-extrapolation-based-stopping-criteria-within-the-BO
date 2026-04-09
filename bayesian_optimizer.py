
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from surrogate_model import SugorrateModel


class BayesianOptimizer:

    def __init__(self,bounds, num_iter, surrogate, acquisition = None, stopping = None ) -> None:
        self.bounds = bounds
        self.num_iter = num_iter
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.stopping = stopping
        self.X = []
        self.y = []

    def initialize(self, objective):
        for _ in range(self.num_iter):
            #x = np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            x = np.random.uniform(self.bounds[0], self.bounds[-1])
            y = objective(x)

            self.X.append(x)
            self.y.append(y)

    def loop(self):
        self.surrogate.fit(self.X, self.y)

        best_y = np.max(self.y)

        #y_pred, y_std = self.surrogate.predict(np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds]))
        # sample_x = np.random.choice(x_range, size=self.num_iter).reshape(-1,1)
        y_pred, y_std = self.surrogate.predict(self.bounds.reshape(-1,1))
        return y_pred, y_std

def black_box_function(x):
    y = np.sin(x) + np.cos(2*x)
    return y

if __name__ == "__main__":
    x_range = np.linspace(-2*np.pi, 2*np.pi, 200)
    surrogate_model = SugorrateModel()
    optimizer = BayesianOptimizer(x_range,15,surrogate_model)
    optimizer.initialize(black_box_function)

    y_pred, y_std = optimizer.loop()
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, black_box_function(x_range), label='Black Box Function')
    plt.scatter(optimizer.X, optimizer.y, color='red', label='Samples')
    plt.plot(x_range, y_pred, color='blue', label='Gaussian Process')
    plt.fill_between(x_range, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)

    plt.xlabel('x')
    plt.ylabel('Black Box Output')
    plt.title('Black Box Function with Gaussian Process Surrogate Model')
    plt.legend()
    plt.show()




