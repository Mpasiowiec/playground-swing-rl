import time
from playground_swing_rl.env.playground_swing import PlaygroundSwingEnv

# Create the environment with rendering enabled
env = PlaygroundSwingEnv(render_mode='human-plots')
data = []
start = time.time()
# Reset the environment to get the initial observation
obs, info = env.reset(options={'theta': -33, 'phi': env.phi_mean, 'psi': env.psi_mean, 'theta_dot':5})


# Run a short demo with random actions
for step in range(int(3/env.dt)):
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    data.append({'theta': obs[0], 'theta_d': obs[1], 'phi': obs[2], 'phi_d': obs[3], 'psi': obs[4], 'psi_d': obs[5]})
end = time.time()
print(end - start)
env.save_gif_from_frames('11.gif')
