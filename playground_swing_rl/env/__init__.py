from gymnasium.envs.registration import register

register(
    id='PlaygroundSwingEnv-v0',
    entry_point='playground_swing_rl.env.playground_swing:PlaygroundSwingEnv',
    max_episode_steps=2000,
    order_enforce=True
)