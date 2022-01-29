import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit


class OptimisticBandit(Bandit):
    """
    class for an optimistic bandit
    """
    def initialize(self, Q):
        self.Q = Q  # estimate of reward


if __name__ == "__main__":
    testbed = [OptimisticBandit(3, Q=20), OptimisticBandit(1, Q=20), OptimisticBandit(5, Q=20)]
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
