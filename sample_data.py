import numpy as np


def sample_data(data_type, num_contexts=None):
  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=0.0)
    opt_rewards, opt_actions = opt_linear

  return dataset, opt_rewards, opt_actions, num_actions, context_dim


def sample_linear_data(num_contexts, dim_context, num_actions, sigma=0.0):
  """Samples data from linearly parameterized arms.

  The reward for context X and arm j is given by X^T beta_j, for some latent
  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly
  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sigma: Standard deviation of the additive noise. Set to zero for no noise.

  Returns:
    data: A [n, d+k] numpy array with the data.
    betas: Latent parameters that determine expected reward for each arm.
    opt: (optimal_rewards, optimal_actions) for all contexts.
  """

  betas = np.random.uniform(-1, 1, (dim_context, num_actions))
  betas /= np.linalg.norm(betas, axis=0)
  contexts = np.random.normal(size=[num_contexts, dim_context])
  rewards = np.dot(contexts, betas)
  opt_actions = np.argmax(rewards, axis=1)
  rewards += np.random.normal(scale=sigma, size=rewards.shape)
  opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
  print(f"betas: {betas}")
  return np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions) 