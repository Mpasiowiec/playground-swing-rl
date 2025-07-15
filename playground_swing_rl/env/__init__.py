from gymnasium.envs.registration import register

register(
    id='PlaygroundSwingEnv-v0',
    entry_point='playground_swing_rl.envs.playground_swing:PlaygroundSwingEnv',
    max_episode_steps=500,
)
