import gymnasium
from playground_swing_rl.env.playground_swing import PlaygroundSwingEnv

env = gymnasium.make("PlaygroundSwingEnv-v0", render_mode='human-plots')
# strategy = PaperModelPolicy(env)
obs, info = env.reset()
ter = False 
while not ter:
    action = env.action_space.sample()
    obs, rewards, ter, trunc, info = env.step(action)

env.unwrapped.save_gif_from_frames('11.gif')