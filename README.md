# Gym Safety

This repository provides unofficial simple environments for research on safety in reinforcement learning.
All environments provide the Openai gym interface and are usable without mujoco.
The goal is to have some easily accessible environments where people can test their algorithms for safe rl.

## Installation
Clone the repository and then type the following in the root directory of it:
```
pip install --user .
```
Installs the environments for the current user.

## Usage
```python
import gym
import gym_safety
env = gym.make('CartSafe-v0')
```

## Environments
The Environments use the model of constrained markov decision processes (CMDPs) where at each step the environment not only returns a reward but also an immediate constraint cost.
The goal is to find an optimal policy while keeping the cumulative constraint costs below a threshold.

The constraint cost is returned inside the info dictionary and can be accessed using
```python
observation, reward, done, info = env.step(action)
c_cost = info['constraint_cost']
```
For more information about the individual environments see the docstrings in the implementation files.

### CartSafe-v0

This is a modification of the original `CartPole-v0` environment provided by openai gym.
In `CartSafe-v0` the goal is to swing up the pendulum from the bottom while staying in a certain region.
For every step outside of the region a constraint cost of 1 is returned.

![CartSafe-v0](gym_safety/assets/cartsafe.png)

### GridNav-v0

This is an unofficial implementation of the environment used by [[1]](#references).
Here the goal is to navigate inside a gridworld without hitting obstacles.
Everytime an obstacle is hit it returns a constraint cost of 1.
The paper solves this by assuming that there already exists a safe baseline policy and then improving this policy without violating the threshold on the cumulative constraint cost.

![GridNav-v0](gym_safety/assets/grid.png)
### MountainCarContinuousSafe-v0

This is a modification of the [original](https://gym.openai.com/envs/MountainCarContinuous-v0/) `MountainCarContinuous` environment from openai gym.
The goal is to drive a cart up a hill by going back and forth and building up momentum.
The constraint is to not go too far to the left (indicated by red line in visualization), every step the cart is too far on the left a cost of 1 is received.
It is a very challenging environment, because a positive reward is only given for reaching the goal.
This means that the exploration strategy needs to find the sparse reward while still obeying the constraints.

![MountainCarContinuousSafe-v0](gym_safety/assets/mountain.png)
## Future Goals
Implement Pybullet Alternatives of the environments used in [[2]](#references) and [[3]](#references):
- Point-Gather
- Point-Circle
- Ant-Gather
- Ant-Circle
- Humanoid-Circle
- HalfCheetah-Safe

This might also happen in the [pybullet-gym](https://github.com/benelot/pybullet-gym) repository.




## References
[1] [A Lyapunov-based Approach to Safe Reinforcement Learning](https://arxiv.org/abs/1805.07708)

[2] [Lyapunov-based Safe Policy Optimization for Continuous Control](https://arxiv.org/abs/1901.10031)

[3] [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
