


class BayesianOptimizer:

    def __init__(self, num_iteraciones, surrogate, acquisition, stopping ) -> None:
        self.num_iteraciones = num_iteraciones
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.stopping = stopping

    def step(self):
        
        pass