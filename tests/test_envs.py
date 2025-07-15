import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from playground_swing_rl.env.playground_swing import PlaygroundSwingEnv

def test_custom_env_compliance():
    # env = gym.make("PlaygroundSwingEnv-v0")
    env = PlaygroundSwingEnv()
    check_env(env)

def test_random_steps():
    # env = gym.make("PlaygroundSwingEnv-v0")
    env = PlaygroundSwingEnv()
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
