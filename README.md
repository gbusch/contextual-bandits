# Contextual Bandits

## Reinforcement Learning

Sutton & Barto: Reinforcement Learning. An Introduction (2018):

> Reinforcement learning is learning what to do — how to map situations to actions — so as to maximize a numerical reward signal. 
> 
> The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. 
> 
> In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. 
> 
> These two characteristics — trial-and-error search and delayed reward — are the two most important distinguishing features of reinforcement learning.


### Reward Signal

On every time step, the reinforcement learning agent receives a single number, called the *reward* from the environment, based on its action.

Goal of the reinforcement learning problem is to maximize the reward on the long term.


### Value Function

While the reward tells what is good in the immediate sense, the value function tells what is good in the long run.

Reward is given after every step, value - however - has to be estimated taking into account previous learning experience.


### Policy

A *policy* is a mapping of the perceived state of the environment to actions that should be taken when in this state.
