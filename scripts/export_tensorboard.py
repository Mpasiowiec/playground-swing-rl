import os
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as e:
    raise SystemExit("tensorboard is required. Install with: pip install tensorboard") from e


def load_scalars_from_event(run_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all scalar tags from a TensorBoard run directory into dataframes.

    Returns a dict: tag -> DataFrame(step, wall_time, value)
    """
    ea = EventAccumulator(run_dir)
    ea.Reload()
    scalars: Dict[str, pd.DataFrame] = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            'step': [e.step for e in events],
            'wall_time': [e.wall_time for e in events],
            'value': [e.value for e in events],
        })
        scalars[tag] = df
    return scalars


def save_run_artifacts(run_name: str, scalars: Dict[str, pd.DataFrame], outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    saved_paths: List[str] = []
    # Save CSVs
    csv_dir = os.path.join(outdir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    for tag, df in scalars.items():
        safe_tag = tag.replace('/', '_')
        csv_path = os.path.join(csv_dir, f"{run_name}__{safe_tag}.csv")
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)
    # Save plots
    plot_dir = os.path.join(outdir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    for tag, df in scalars.items():
        if df.empty:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(df['step'], df['value'], label=tag)
        plt.xlabel('Step')
        plt.ylabel(tag)
        plt.title(f"{run_name}: {tag}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_tag = tag.replace('/', '_')
        plot_path = os.path.join(plot_dir, f"{run_name}__{safe_tag}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        saved_paths.append(plot_path)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description='Export TensorBoard scalars (CSV + PNG plots).')
    parser.add_argument('--logdir', type=str, default='models/logs', help='Directory containing TensorBoard runs')
    parser.add_argument('--outdir', type=str, default='reports', help='Output directory for CSVs and plots')
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        raise SystemExit(f"Logdir not found: {args.logdir}")

    run_dirs = [d for d in os.listdir(args.logdir) if os.path.isdir(os.path.join(args.logdir, d))]
    if not run_dirs:
        raise SystemExit(f"No run directories found in {args.logdir}")

    print(f"Found runs: {run_dirs}")
    for run in run_dirs:
        run_path = os.path.join(args.logdir, run)
        try:
            scalars = load_scalars_from_event(run_path)
            if not scalars:
                print(f"No scalars in {run}")
                continue
            saved = save_run_artifacts(run, scalars, args.outdir)
            print(f"Saved {len(saved)} files for run {run} -> {args.outdir}")
        except Exception as e:
            print(f"Failed to export {run}: {e}")


if __name__ == '__main__':
    main()


