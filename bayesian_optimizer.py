
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from acquisition_function import acquisition_ucb
from surrogate_model import SurrogateModel


class BayesianOptimizer:

    def __init__(self,bounds,num_samples,num_iter, surrogate, acquisition, stopping = None ) -> None:
        self.bounds = bounds
        self.num_iter = num_iter
        self.num_samples = num_samples
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.stopping = stopping
        self.X = []
        self.y = []

    def initialize(self, objective):
        for _ in range(self.num_samples):
            #x = np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds])
            x = np.random.uniform(self.bounds[0], self.bounds[-1])
            y = objective(x)

            self.X = np.append(self.X, x)
            self.y = np.append(self.y, y)
    
    def plot(self, y_pred, y_std,new_x,new_y,i):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.bounds, self.acquisition(self.bounds), label='Black Box Function')
        ax.scatter(self.X, self.y, color='red', label='Samples')
        ax.scatter(new_x, new_y, color='blue', label='New point')
        ax.plot(self.bounds, y_pred, color='blue', label='Gaussian Process')
        ax.fill_between(self.bounds, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
        ax.set(title=f'Black Box Function with Gaussian Process Surrogate Model Loop Nº{i}', ylabel='Black Box Output', xlabel='x')
        plt.show()


    def loop(self):
        for i in range(self.num_iter):
            self.surrogate.fit(self.X, self.y)

            #y_pred, y_std = self.surrogate.predict(np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds]))
            # sample_x = np.random.choice(x_range, size=self.num_iter).reshape(-1,1)

            y_pred, y_std = self.surrogate.predict(self.bounds.reshape(-1,1))
            ucb = self.acquisition(y_pred, y_std)

            new_x = self.bounds[np.argmax(ucb)]
            new_y = self.acquisition(new_x)
            self.X = np.append(self.X, new_x)
            self.y = np.append(self.y, new_y)
            self.plot(y_pred,y_std,new_x,new_y,i)



def black_box_function(x):
    y = np.sin(x) + np.cos(2*x)
    return y

if __name__ == "__main__":
    x_range = np.linspace(-2*np.pi, 2*np.pi, 200)
    surrogate_model = SurrogateModel()
    optimizer = BayesianOptimizer(x_range,5,15,surrogate_model,acquisition_ucb)
    optimizer.initialize(black_box_function)
    optimizer.loop()




