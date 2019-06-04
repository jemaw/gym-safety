from gym.envs.registration import register

register(
    id='cart_safe-v0',
    entry_point='gym_safety.envs:CartSafeEnv',
)
