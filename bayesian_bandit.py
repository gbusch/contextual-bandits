import numpy as np
from scipy.stats import beta


class BayesianBandit:
    """
    class for one bayesian bandit
    """
    def __init__(self, q, prior=(1., 1.), **kwargs):
        self.q = q  # mean of underlying probability
        self.N = 0  # number of previous samplings
        self.s = 0  # number of previous successes
        self.prior = prior

    def pull(self):
        return np.random.random() < self.q

    def update(self, R):
        self.N += 1
        self.s += R

    def sample(self):
        return np.random.beta(self.prior[0] + self.s, self.prior[1] + self.N - self.s)
    
    def posterior(self):
        x = np.arange(-0.1, 1.1, 0.01)
        return x, beta.pdf(x, self.prior[0] + self.s, self.prior[1] + self.N - self.s)
