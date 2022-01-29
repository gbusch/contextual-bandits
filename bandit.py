import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    testbed = [Bandit(3), Bandit(1), Bandit(5)]
    N_TRIALS = 50

    rewards = np.ndarray((N_TRIALS +1, len(testbed)))

    estimated_rewards = [bandit.sample() for bandit in testbed]
    rewards[0, :] = estimated_rewards
    for n in range(N_TRIALS):
        bandit = testbed[np.argmax(estimated_rewards)]
        R = bandit.pull()
        bandit.update(R)
        estimated_rewards = [bandit.sample() for bandit in testbed]
        rewards[n + 1, :] = estimated_rewards

    plt.figure()
    for b in range(rewards.shape[1]):
        plt.plot(rewards[:, b], label=f'Bandit {b+1}')
    plt.legend()
    plt.show()
