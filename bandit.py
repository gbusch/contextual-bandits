import numpy as np


class Bandit:
    """
    class for one bandit
    """
    def __init__(self, q, **kwargs):
        self.q = q  # mean of underlying reward distribution
        self.n = 0  # number of previous samplings
        self.initialize(**kwargs)

    def initialize(self):
        self.Q = 0  # estimate of reward

    def pull(self):
        value = np.random.randn() + self.q
        return value if value > 0 else 0

    def update(self, R):
        self.n += 1
        self.Q += (R - self.Q) / self.n

    def sample(self):
        return self.Q
