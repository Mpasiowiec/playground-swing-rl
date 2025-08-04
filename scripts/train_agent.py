import os
import argparse
import datetime

import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id 

from playground_swing_rl.env.playground_swing import PlaygroundSwingEnv

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train():

    run_id = get_latest_run_id('models', f"{args.gymenv}_{args.sb3_algo}")
    folder_path = os.path.join(model_dir, f"{args.gymenv}_{args.sb3_algo}_{run_id+1}")
    os.makedirs(folder_path, exist_ok=True)
    
    model = sb3_class(
        'MlpPolicy',
        env,
        device='cpu',
        tensorboard_log=log_dir,
        verbose=0
        )
 
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=15, # min number of steps before checking for improvement
        verbose=0
        )

    eval_callback = EvalCallback(
        env, 
        eval_freq=10_000, # how often perform evaluation
        callback_after_eval=stop_train_callback, 
        verbose=0, 
        best_model_save_path=folder_path,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=folder_path,
        name_prefix='checkpoint'
    )

    
    model.learn(
        total_timesteps=int(1e10),
        tb_log_name=f"{args.gymenv}_{args.sb3_algo}",
        callback=[eval_callback, checkpoint_callback],
        )

def test(model_name):        
    model = sb3_class.load(
        os.path.join(model_dir, f"{args.gymenv}_{args.sb3_algo}_{model_name}"),
        env=env
        )

    obs = env.reset(options={'theta':0, 'theta_dot':0})[0]   
    
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--gymenv', help='Gymnasium environment i.e. Humanoid-v4', default='PlaygroundSwingEnv-v0')
    parser.add_argument('--sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, PPO, SAC, TD3', default='A2C')    
    parser.add_argument('--test', help='Test mode', action='store_true')
    parser.add_argument('--test_model', help='Which model to test', default='1/best_model')
    args = parser.parse_args()

    sb3_class = getattr(stable_baselines3, args.sb3_algo)

    if args.test:
        env = gym.make(args.gymenv, render_mode='human-plots')
        # recording_dir = f"vid/vid_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # os.makedirs(recording_dir)
        # env = gym.wrappers.RecordVideo(env, video_folder=recording_dir, episode_trigger=lambda x: True, name_prefix="test")
        test(args.test_model)
    else:
        env = gym.make(args.gymenv)
        env = Monitor(env)
        train()
    env.close()