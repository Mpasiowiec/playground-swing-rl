import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def read_metric_df(csvdir: str, run_prefix: str, tag_key: str) -> pd.DataFrame:
    candidates = [
        os.path.join(csvdir, f"{run_prefix}__{tag_key}.csv"),
        os.path.join(csvdir, f"{run_prefix}_{tag_key}.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.isfile(p)), None)
    if csv_path is None:
        raise FileNotFoundError(f"CSV not found. Looked for: {candidates}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    if 'step' not in df.columns or 'value' not in df.columns:
        raise KeyError(f"CSV must contain 'Step' and 'Value' columns. Found: {df.columns}")
    return df[['step', 'value']]


def plot_overlay(csvdir: str, run_prefixes: list[str], tag_key: str, title: str, out_path: str):
    plt.figure(figsize=(8, 4))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for idx, run in enumerate(run_prefixes):
        try:
            df = read_metric_df(csvdir, run, tag_key)
        except Exception:
            continue
        suffix = run.rsplit('_', 1)[-1] if '_' in run else run
        label = f"A2C {suffix}"
        plt.plot(df['step'], df['value'], label=label, linewidth=2.0, color=colors[idx % len(colors)])
    plt.xlabel('Step')
    plt.ylabel(tag_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training and eval rewards from CSVs')
    parser.add_argument('--csvdir', type=str, default='reports')
    parser.add_argument('--outdir', type=str, default='reports/plots')
    parser.add_argument('--runs', nargs='+', default=['PlaygroundSwingEnv-v0_A2C_1', 'PlaygroundSwingEnv-v0_A2C_2', 'PlaygroundSwingEnv-v0_A2C_3'], help='Run prefixes to overlay')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Training mean reward (rollout)
    train_key = 'ep_rew_mean'
    train_out = os.path.join(args.outdir, f"PlaygroundSwingEnv-v0_A2C_1__rollout_ep_rew_mean.png")
    plot_overlay(args.csvdir, args.runs, train_key, 'Training: episode mean reward (A2C1/2/3)', train_out)

    # Evaluation mean reward
    eval_key = 'eval_mean_reward'
    eval_out = os.path.join(args.outdir, f"PlaygroundSwingEnv-v0_A2C_1__eval_mean_reward.png")
    plot_overlay(args.csvdir, args.runs, eval_key, 'Evaluation: mean reward (A2C1/2/3)', eval_out)

    print(f"Saved plots:\n - {train_out}\n - {eval_out}")


if __name__ == '__main__':
    main()


