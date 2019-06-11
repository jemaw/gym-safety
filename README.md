# Gym Safety

This repository provides unofficial simple environments for reseaerch on safety in reinforcement learning.
All environments provide the Openai gym interface.

## Installation
Clone the repository and then type the following in the root directory of it:
```
pip install --user .
```
Installs the environments for the current user.

## Environments
The Environments use the model of constrained markov decision processes (CMDPs) where at each step the environment not only returns a reward but also an immediate constraint cost.
The goal is to find an optimal policy while keeping the cumulative constraint costs below a threshold.

### CartSafe-v0

This is a modification of the original `CartPole-v0` environment provided by openai gym.
In `CartSafe-v0` the goal is to swing up the pendulum from the bottom while staying in a certain region.
For every step outside of the region a constraint cost of 1 is returned.

### GridNav-v0

This is an unofficial implementation of the environment used by [A Lyapunov-based Approach to Safe Reinforcement Learning][1].
Here the goal is to navigate inside a gridworld without hitting obstacles.
Everytime an obstacle is hit it returns a constraint cost of 1.
The paper solves this by assuming that there already exists a safe baseline policy and then improving this policy without violating the threshold on the cumulative constraint cost.

[1]: https://arxiv.org/abs/1805.07708
