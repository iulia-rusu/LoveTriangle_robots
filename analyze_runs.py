"""
Analyze every completed DQN training run under outputs/dqn_runs (or --root):

- Discovers runs that actually finished (have both training_metrics.csv and
  policies/final_policy.pt — train.py only writes final_policy.pt once the
  full episode loop completes, so in-progress runs are skipped automatically).
- Computes summary metrics per run (mean/best reward, success rate, episode
  duration) from training_metrics.csv.
- Parses each run's swept hyperparameters out of run_config.json's
  cli_args.overrides (e.g. "training.lr=0.0003" -> {"training.lr": 0.0003}).
- Writes a leaderboard CSV and prints the top N runs.
- Plots the best run's reward curve against all the others.
- Plots the chosen metric against each swept variable found across the runs,
  to show how each one affects performance.

Usage:
    python analyze_runs.py
    python analyze_runs.py --root outputs/dqn_runs --metric success_rate_last100 --top-n 10
    python analyze_runs.py --window 200 --out-dir outputs/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dqn_common import parse_scalar


def find_completed_runs(root: Path) -> list[Path]:
    """Return the directory of every run under root that has both a
    run_config.json and a policies/final_policy.pt (i.e. training actually
    finished, not just checkpointed mid-way).
    """
    run_dirs = []
    for run_config_path in root.glob("**/run_config.json"):
        run_dir = run_config_path.parent
        if (run_dir / "policies" / "final_policy.pt").exists() and (run_dir / "training_metrics.csv").exists():
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    parsed = {}
    for item in overrides or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = parse_scalar(value.strip())
    return parsed


def compute_metrics(metrics_csv: Path, window: int) -> dict[str, Any]:
    df = pd.read_csv(metrics_csv)
    n_episodes = len(df)
    tail = df.tail(window) if n_episodes > 0 else df

    def success_rate(frame: pd.DataFrame) -> float:
        if len(frame) == 0:
            return float("nan")
        return float((frame["done_reason"] == "green_collision").mean())

    return {
        "n_episodes": n_episodes,
        "mean_reward_all": float(df["reward"].mean()) if n_episodes else float("nan"),
        "mean_reward_last": float(tail["reward"].mean()) if len(tail) else float("nan"),
        "best_reward": float(df["reward"].max()) if n_episodes else float("nan"),
        "mean_duration_last": float(tail["duration"].mean()) if len(tail) else float("nan"),
        "success_rate_all": success_rate(df),
        "success_rate_last": success_rate(tail),
    }


def build_summary(run_dirs: list[Path], window: int) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        run_config = json.loads((run_dir / "run_config.json").read_text())
        overrides = parse_overrides(run_config.get("cli_args", {}).get("overrides", []))
        resolved = run_config.get("resolved_training", {})

        row = {
            "run_dir": str(run_dir),
            "condition": resolved.get("condition"),
            "seed": resolved.get("seed"),
            "n_actions": resolved.get("n_actions"),
        }
        row.update(overrides)
        row.update(compute_metrics(run_dir / "training_metrics.csv", window))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def smoothed(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for a moving-average curve, or the raw series if it's
    shorter than window.
    """
    if len(values) < window:
        return np.arange(len(values)), values
    moving = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, window - 1 + len(moving)), moving


def plot_best_vs_others(df: pd.DataFrame, metric: str, window: int, out_path: Path) -> None:
    if df.empty:
        return
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, row in df.iterrows():
        csv_path = Path(row["run_dir"]) / "training_metrics.csv"
        rewards = pd.read_csv(csv_path)["reward"].to_numpy()
        x, y = smoothed(rewards, window)
        if idx == best_idx:
            continue
        ax.plot(x, y, color="lightgray", linewidth=1, zorder=1)

    csv_path = Path(best_row["run_dir"]) / "training_metrics.csv"
    rewards = pd.read_csv(csv_path)["reward"].to_numpy()
    x, y = smoothed(rewards, window)
    ax.plot(x, y, color="crimson", linewidth=2, zorder=2, label=f"best ({metric}={best_row[metric]:.2f})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({window}-episode moving average)")
    ax.set_title(f"Best run vs. all others\nbest run: {Path(best_row['run_dir']).name}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def known_non_variable_columns() -> set[str]:
    return {
        "run_dir", "condition", "seed", "n_actions",
        "n_episodes", "mean_reward_all", "mean_reward_last", "best_reward",
        "mean_duration_last", "success_rate_all", "success_rate_last",
    }


def plot_metric_vs_variables(df: pd.DataFrame, metric: str, out_dir: Path) -> list[str]:
    """One scatter plot per swept variable (every column that isn't a fixed
    metadata/metric column), showing how that variable relates to metric.
    Only plots variables with at least 2 distinct non-null values across the
    discovered runs. Returns the list of variable names actually plotted.
    """
    variable_cols = [c for c in df.columns if c not in known_non_variable_columns()]
    plotted = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in variable_cols:
        sub = df[[col, metric]].dropna()
        if sub[col].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        if pd.api.types.is_numeric_dtype(sub[col]):
            ax.scatter(sub[col], sub[metric], alpha=0.7)
            ax.set_xlabel(col)
        else:
            categories = sorted(sub[col].astype(str).unique())
            positions = {c: i for i, c in enumerate(categories)}
            xs = sub[col].astype(str).map(positions)
            ax.scatter(xs, sub[metric], alpha=0.7)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=30, ha="right")
            ax.set_xlabel(col)

        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {col}")
        fig.tight_layout()
        safe_name = col.replace("/", "_").replace("\\", "_").replace(".", "_")
        fig.savefig(out_dir / f"{safe_name}.png", dpi=200)
        plt.close(fig)
        plotted.append(col)

    return plotted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed DQN training runs.")
    parser.add_argument("--root", type=str, default="outputs/dqn_runs")
    parser.add_argument("--out-dir", type=str, default="outputs/analysis")
    parser.add_argument("--metric", type=str, default="success_rate_last",
                         choices=["mean_reward_all", "mean_reward_last", "best_reward",
                                  "success_rate_all", "success_rate_last"],
                         help="Metric used to rank runs and pick the 'best' one.")
    parser.add_argument("--window", type=int, default=100,
                         help="Episode window for moving averages / 'last window' metrics.")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)

    run_dirs = find_completed_runs(root)
    if not run_dirs:
        print(f"No completed runs found under {root} (need both training_metrics.csv and policies/final_policy.pt).")
        return

    print(f"Found {len(run_dirs)} completed run(s) under {root}.")
    df = build_summary(run_dirs, args.window)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    df.sort_values(args.metric, ascending=False).to_csv(summary_path, index=False)
    print(f"Wrote leaderboard to {summary_path}")

    top = df.sort_values(args.metric, ascending=False).head(args.top_n)
    display_cols = ["run_dir", args.metric, "mean_reward_last", "success_rate_last", "best_reward", "n_episodes"]
    display_cols = [c for c in display_cols if c in top.columns]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(f"\nTop {len(top)} run(s) by {args.metric}:")
        print(top[display_cols].to_string(index=False))

    best_plot_path = out_dir / "best_vs_others.png"
    plot_best_vs_others(df, args.metric, args.window, best_plot_path)
    print(f"\nSaved best-vs-others training curve to {best_plot_path}")

    variables_dir = out_dir / "variables"
    plotted = plot_metric_vs_variables(df, args.metric, variables_dir)
    if plotted:
        print(f"Saved {len(plotted)} variable-sensitivity plot(s) to {variables_dir}: {', '.join(plotted)}")
    else:
        print("No swept variable had 2+ distinct values across the discovered runs — skipped sensitivity plots.")

    best_row = df.sort_values(args.metric, ascending=False).iloc[0]
    print(f"\nBest run by {args.metric}: {best_row['run_dir']}")
    print(f"  policy: {Path(best_row['run_dir']) / 'policies' / 'final_policy.pt'}")
    print(f"  rollout figures (from training): {Path(best_row['run_dir']) / 'figures'}")


if __name__ == "__main__":
    # Only force the non-interactive Agg backend for CLI runs (this script
    # only ever saves figures, never shows them) — setting it at import time
    # instead would poison the backend for anything that imports this module
    # afterwards, e.g. analyze_runs.ipynb, where plt.show() needs an inline
    # backend to actually render anything.
    import matplotlib
    matplotlib.use("Agg")
    main()
