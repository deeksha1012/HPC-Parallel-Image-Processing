"""
Performance visualization charts for HPC image processing benchmarks.

Generates Matplotlib charts for:
1. Execution time comparison across sequential and parallel configurations.
2. Parallel speedup curve compared against ideal linear speedup.
3. Parallel efficiency curve compared against ideal 100% scaling.

Uses the non-interactive 'Agg' backend for thread and process safety.
Always close figures after use (plt.close(fig)) to prevent memory leaks.
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI thread conflicts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


def plot_execution_time(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a bar chart of mean execution times with standard deviation error bars.

    Args:
        df (pd.DataFrame): Benchmark results DataFrame containing 'mode',
                           'workers', 'mean_time_s', and 'std_time_s'.

    Returns:
        plt.Figure: Matplotlib figure containing the execution time bar chart.
    """
    labels = [
        f"{int(w)} workers" if m == "Parallel" else "Sequential"
        for m, w in zip(df["mode"], df["workers"])
    ]
    x_positions = np.arange(len(df))
    means = df["mean_time_s"].values
    stds = df["std_time_s"].values

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Bar chart with error bars representing trial variance
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color="#1f77b4",
        edgecolor="#08306b",
        alpha=0.85,
        label="Mean Time",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Processing Configuration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Execution Time by Configuration", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    return fig


def plot_speedup(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a line chart comparing measured speedup against ideal linear speedup.

    Args:
        df (pd.DataFrame): Benchmark results DataFrame.

    Returns:
        plt.Figure: Matplotlib figure containing the speedup chart.
    """
    par_df = df[df["mode"] == "Parallel"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if par_df.empty:
        ax.text(0.5, 0.5, "No parallel configurations tested", ha="center", va="center")
        ax.set_title("Parallel Speedup vs. Worker Processes")
        return fig

    workers = par_df["workers"].astype(int).values
    speedup = par_df["speedup"].values

    # Measured speedup line
    ax.plot(
        workers,
        speedup,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#1f77b4",
        label="Measured Speedup",
    )

    # Ideal linear speedup line (S = p)
    ax.plot(
        workers,
        workers,
        linestyle="--",
        linewidth=1.8,
        color="#e6550d",
        label="Ideal Speedup (Linear: S = p)",
    )

    ax.set_xticks(workers)
    ax.set_xlabel("Number of Worker Processes (Cores)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup Factor (S)", fontsize=11, fontweight="bold")
    ax.set_title("Parallel Speedup vs. Worker Processes", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    return fig


def plot_efficiency(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a line chart comparing parallel efficiency against the 100% ideal baseline.

    Args:
        df (pd.DataFrame): Benchmark results DataFrame.

    Returns:
        plt.Figure: Matplotlib figure containing the efficiency chart.
    """
    par_df = df[df["mode"] == "Parallel"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if par_df.empty:
        ax.text(0.5, 0.5, "No parallel configurations tested", ha="center", va="center")
        ax.set_title("Parallel Efficiency vs. Worker Processes")
        return fig

    workers = par_df["workers"].astype(int).values
    efficiency_pct = par_df["efficiency"].values * 100.0

    # Measured efficiency curve
    ax.plot(
        workers,
        efficiency_pct,
        marker="s",
        linewidth=2,
        markersize=6,
        color="#2ca02c",
        label="Measured Efficiency",
    )

    # Ideal 100% efficiency reference line
    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=1.8,
        color="#e6550d",
        label="Ideal Efficiency (100%)",
    )

    ax.set_xticks(workers)
    ax.set_xlabel("Number of Worker Processes (Cores)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Parallel Efficiency (%)", fontsize=11, fontweight="bold")
    ax.set_title("Parallel Efficiency vs. Worker Processes", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    return fig


def main() -> None:
    """Command-line interface to render and export benchmark figures to PNG."""
    parser = argparse.ArgumentParser(
        description="Generate and export benchmark performance charts."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS_DIR / "benchmark_results.csv"),
        help="Path to benchmark_results.csv (default: results/benchmark_results.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "plots"),
        help="Directory to save generated plot images (default: results/plots)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: Benchmark file '{csv_path}' not found.")
        print("Please run the benchmark first using: python -m src.benchmark")
        return

    df = pd.read_csv(csv_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render and save Execution Time chart
    fig_time = plot_execution_time(df)
    time_file = out_dir / "execution_time.png"
    fig_time.savefig(time_file, dpi=150, bbox_inches="tight")
    plt.close(fig_time)
    print(f"Saved: {time_file}")

    # Render and save Speedup chart
    fig_speedup = plot_speedup(df)
    speedup_file = out_dir / "speedup.png"
    fig_speedup.savefig(speedup_file, dpi=150, bbox_inches="tight")
    plt.close(fig_speedup)
    print(f"Saved: {speedup_file}")

    # Render and save Efficiency chart
    fig_eff = plot_efficiency(df)
    eff_file = out_dir / "efficiency.png"
    fig_eff.savefig(eff_file, dpi=150, bbox_inches="tight")
    plt.close(fig_eff)
    print(f"Saved: {eff_file}")


if __name__ == "__main__":
    main()
