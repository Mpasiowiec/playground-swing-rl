from playground_swing_rl.env.playground_swing import PlaygroundSwingEnv
from gymnasium.wrappers import TimeLimit
import pandas as pd
import sys


data = []

env = TimeLimit(PlaygroundSwingEnv(render_mode='human'), max_episode_steps=100)
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    data.append({'theta': obs[0],'theta_dot': obs[1],'phi': obs[2],'phi_d': obs[3],'psi': obs[4],'psi_d': obs[5]})
    if not env.observation_space.contains(obs): 
        print(f'State outside of the observation space. {obs}')
        # break
        
df = pd.DataFrame(data)