from gym.envs.registration import register

register(
    id='CartSafe-v0',
    entry_point='gym_safety.envs:CartSafeEnv',
    max_episode_steps=300,
    reward_threshold=520.0,
)

register(
    id='GridNav-v0',
    entry_point='gym_safety.envs:GridNavigationEnv',
    kwargs={'gridsize': 25},
    max_episode_steps=200
)

register(
    id='GridNav-v1',
    entry_point='gym_safety.envs:GridNavigationEnv',
    kwargs={'gridsize': 60},
    max_episode_steps=200
)

register(
    id='MountainCarContinuousSafe-v0',
    entry_point='gym_safety.envs:Continuous_MountainCarSafeEnv',
    reward_threshold=90,
    max_episode_steps=999,
)
