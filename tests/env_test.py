import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import playground_swing_rl.env.playground_swing  
import numpy as np

def test_custom_env_compliance():
    env = gym.make("PlaygroundSwingEnv-v0")
    check_env(env.unwrapped)

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


def test_action_clipping_and_space_compliance():
    env = gym.make("PlaygroundSwingEnv-v0")
    obs, info = env.reset()
    # Intentionally oversized action to test clipping and stability
    big_action = np.array([10.0, -10.0], dtype=np.float32)
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(big_action)
        assert env.observation_space.contains(obs), "Observation went out of bounds after large action."
        if terminated or truncated:
            break
    env.close()


def test_termination_eventually_occurs():
    env = gym.make("PlaygroundSwingEnv-v0")
    obs, info = env.reset()
    terminated = False
    truncated = False
    # Environment uses internal time and should terminate within a reasonable number of steps
    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated, "Episode did not end within 2000 steps."
    env.close()


def test_render_rgb_array_modes():
    # Verify that rgb_array modes return a NumPy array frame without raising
    for mode in ["rgb_array", "rgb_array_plots"]:
        env = gym.make("PlaygroundSwingEnv-v0", render_mode=mode)
        obs, info = env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray), f"Render did not return an array for mode {mode}."
        assert frame.ndim == 3 and frame.shape[2] == 3, "Render frame should be HxWx3."
        env.close()


def test_seeding_reproducibility():
    # Two environments with the same seed should produce identical initial observations
    env1 = gym.make("PlaygroundSwingEnv-v0")
    env2 = gym.make("PlaygroundSwingEnv-v0")
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_allclose(obs1, obs2, rtol=0, atol=1e-8)
    env1.close()
    env2.close()


def test_spaces_shapes_and_dtypes():
    env = gym.make("PlaygroundSwingEnv-v0")
    # Action space is 2D continuous
    assert env.action_space.shape == (2,), "Action space shape should be (2,)"
    # Observation space is 6D continuous
    assert env.observation_space.shape == (6,), "Observation space shape should be (6,)"
    obs, _ = env.reset()
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    env.close()