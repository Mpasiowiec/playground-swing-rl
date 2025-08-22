import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(csvdir: str, run_prefix: str, tag_key: str, out_path: str):
    # Support both flat CSV filenames (e.g., Run_ep_rew_mean.csv) and previous scheme
    candidates = [
        os.path.join(csvdir, f"{run_prefix}__{tag_key}.csv"),
        os.path.join(csvdir, f"{run_prefix}_{tag_key}.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.isfile(p)), None)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found. Looked for: {candidates}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    # Normalize columns for case/space differences from various exporters
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    if 'step' not in df.columns or 'value' not in df.columns:
        raise KeyError(f"CSV must contain 'Step' and 'Value' columns. Found: {df.columns}")
    plt.figure(figsize=(8, 4))
    plt.plot(df['step'], df['value'])
    plt.xlabel('Step')
    plt.ylabel(tag_key)
    plt.title(f"{run_prefix}: {tag_key}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training and eval rewards from CSVs')
    parser.add_argument('--csvdir', type=str, default='reports/csv')
    parser.add_argument('--outdir', type=str, default='reports/plots')
    parser.add_argument('--run', type=str, default='PlaygroundSwingEnv-v0_A2C_1', help='Run prefix (matches CSV file prefix)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Training mean reward (rollout)
    train_key = 'ep_rew_mean'
    train_out = os.path.join(args.outdir, f"{args.run}__rollout_ep_rew_mean.png")
    plot_metric(args.csvdir, args.run, train_key, train_out)

    # Evaluation mean reward
    eval_key = 'eval_mean_reward'
    eval_out = os.path.join(args.outdir, f"{args.run}__eval_mean_reward.png")
    plot_metric(args.csvdir, args.run, eval_key, eval_out)

    print(f"Saved plots:\n - {train_out}\n - {eval_out}")


if __name__ == '__main__':
    main()


