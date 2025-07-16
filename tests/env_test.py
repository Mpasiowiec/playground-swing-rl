import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import playground_swing_rl.env.playground_swing  

def test_custom_env_compliance():
    env = gym.make("PlaygroundSwingEnv-v0")
    check_env(env)

def test_random_steps():
    env = gym.make("PlaygroundSwingEnv-v0")
    obs, info = env.reset()
    for _ in range(101):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.observation_space.contains(obs), f'State outside of the observation space. {obs}'
        
        if terminated or truncated:
            obs, info = env.reset()
    env.close()