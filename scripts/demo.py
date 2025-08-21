import gymnasium
import argparse
from playground_swing_rl.env.playground_swing import PlaygroundSwingEnv
from reference_policies.swing_strategies import NonePolicy, RandomPolicy, FFMPolicy
from gymnasium.wrappers import RecordVideo


if __name__ == '__main__':

    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Show different swinging strategies.')
    parser.add_argument('--strategy', type=str, default="ffm",
                        help='Choose the swinging strategy: e.g. "none", "random", "ffm", "trained".')
    parser.add_argument('--sb3_algo', type=str, default='A2C',
                        help='StableBaseline3 RL algorithm, e.g. A2C, DDPG, PPO, SAC, TD3')
    parser.add_argument('--model_dir', type=str,
                        default='models\\PlaygroundSwingEnv-v0_A2C_1\\best_model.zip',
                        help='Directory of trained model.')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run (default is 1)')
    parser.add_argument('--render', type=str, default="rgb_array_plots",
                        help='Render option for environment: e.g. "human", "human_plots", "rgb_array", "rgb_array_plots".')
    parser.add_argument('--save', type=str, default='videos/',
                        help='Folder to save the animation mp4 files.')
    parser.add_argument('--random_init_angle', action='store_true',
                        help='If set, initializes angle randomly between -π/2 and π/2 radians; '
                             'otherwise initializes at a fixed -33 degrees.')

    args = parser.parse_args()

    # Create environment with specified render mode
    env = gymnasium.make("PlaygroundSwingEnv-v0", render_mode=args.render)

    # Wrap environment to record videos, saving to specified folder with custom prefix
    # Recording every episode by default (episode_trigger=lambda e: True)
    env = RecordVideo(env,
                      video_folder=args.save,
                      name_prefix=f'{args.strategy}-video',
                      episode_trigger=lambda e: True,
                      disable_logger=True)

    # Select the swinging strategy based on user argument
    if args.strategy == "none":
        strategy = NonePolicy()
    elif args.strategy == "random":
        strategy = RandomPolicy()
    elif args.strategy == "ffm":
        strategy = FFMPolicy(env)
    elif args.strategy == "trained":
        import stable_baselines3
        # Dynamically load chosen stable-baselines3 algorithm class and load model from file
        sb3_class = getattr(stable_baselines3, args.sb3_algo)
        strategy = sb3_class.load(args.model_dir, env=env)
        print(f"Model loaded: {args.sb3_algo}, from {args.model_dir}")
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Run the specified number of episodes
    for episode in range(args.episodes):
        # Initialize environment angle: random if flag set, otherwise fixed -33 degrees
        options = None if args.random_init_angle else {'theta': 0}
        obs, info = env.reset(options=options)

        done = False
        # Step through the episode until termination
        while not done:
            action, _ = strategy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    # Close the environment and properly finalize recordings
    env.close()
