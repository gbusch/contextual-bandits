import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit


class OptimisticBandit(Bandit):
    """
    class for an optimistic bandit
    """
    def initialize(self, Q):
        self.Q = Q  # estimate of reward
