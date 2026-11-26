"""
Publication-Quality Visualization Routines for Module 11 Parallel Performance Optimization.

Generates Matplotlib figures comparing:
1. 3-Way Execution Times (Sequential vs. Original Parallel vs. Optimized Parallel)
2. Speedup Curves (Original vs. Optimized vs. Ideal Linear Scaling)
3. Parallel Efficiency Decay Curves
4. Throughput (Images/sec and Megapixels/sec)
5. Workload Scalability & Parallel Performance Crossover Points
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_three_way_execution_time(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a grouped bar chart comparing mean execution times across
    Sequential, Original Parallel, and Optimized Parallel pipelines.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)

    seq_rows = df[df["mode"] == "Sequential"]
    seq_time = float(seq_rows["mean_time_s"].iloc[0]) if not seq_rows.empty else 0.0

    orig_df = df[df["mode"] == "Original Parallel"].sort_values("workers")
    opt_df = df[df["mode"] == "Optimized Parallel"].sort_values("workers")

    workers = sorted(list(set(orig_df["workers"].tolist() + opt_df["workers"].tolist())))
    if not workers:
        ax.text(0.5, 0.5, "No parallel data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    x = np.arange(len(workers))
    width = 0.35

    orig_times = [
        float(orig_df[orig_df["workers"] == w]["mean_time_s"].iloc[0])
        if not orig_df[orig_df["workers"] == w].empty else 0.0
        for w in workers
    ]
    opt_times = [
        float(opt_df[opt_df["workers"] == w]["mean_time_s"].iloc[0])
        if not opt_df[opt_df["workers"] == w].empty else 0.0
        for w in workers
    ]

    rects1 = ax.bar(x - width / 2, orig_times, width, label="Original Parallel (Pipes)", color="#f97316", edgecolor="#c2410c")
    rects2 = ax.bar(x + width / 2, opt_times, width, label="Optimized Parallel (SharedMemory)", color="#10b981", edgecolor="#047857")

    # Horizontal baseline for sequential
    if seq_time > 0:
        ax.axhline(seq_time, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Sequential Baseline ({seq_time:.3f}s)")

    ax.set_title("Execution Time Comparison: Sequential vs. Original vs. Optimized", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Worker Process Allocation", fontsize=10)
    ax.set_ylabel("Mean Execution Time (seconds)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w} Workers" for w in workers])
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.legend(loc="upper right", fontsize=8.5)

    # Value annotations on top of bars
    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}s",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    fig.tight_layout()
    return fig


def plot_speedup_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a comparative speedup curve contrasting Original Parallel Speedup
    against Optimized Parallel Speedup and Ideal Linear Scaling (S = p).
    """
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=150)

    orig_df = df[df["mode"] == "Original Parallel"].sort_values("workers")
    opt_df = df[df["mode"] == "Optimized Parallel"].sort_values("workers")

    workers = sorted(list(set(orig_df["workers"].tolist() + opt_df["workers"].tolist())))
    if not workers:
        ax.text(0.5, 0.5, "No parallel data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    # Plot Ideal Linear Speedup
    all_w = [1] + workers
    ax.plot(all_w, all_w, linestyle=":", color="#64748b", linewidth=1.5, label="Ideal Linear Speedup (S = p)")
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Sequential Baseline (1.0x)")

    # Plot Original Speedup
    if not orig_df.empty:
        w_orig = [1] + orig_df["workers"].tolist()
        s_orig = [1.0] + orig_df["speedup"].tolist()
        ax.plot(w_orig, s_orig, marker="s", color="#f97316", linewidth=2.0, label="Original Parallel Speedup")

    # Plot Optimized Speedup
    if not opt_df.empty:
        w_opt = [1] + opt_df["workers"].tolist()
        s_opt = [1.0] + opt_df["speedup"].tolist()
        ax.plot(w_opt, s_opt, marker="o", color="#10b981", linewidth=2.2, label="Optimized Parallel (SharedMemory)")

    ax.set_title("Parallel Speedup: Original vs. Optimized vs. Ideal Scaling", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Worker Process Count", fontsize=10)
    ax.set_ylabel("Speedup Factor (Sp = T1 / Tp)", fontsize=10)
    ax.set_xticks(all_w)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    return fig


def plot_efficiency_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Generate parallel efficiency comparison curves (Original vs. Optimized).
    """
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)

    orig_df = df[df["mode"] == "Original Parallel"].sort_values("workers")
    opt_df = df[df["mode"] == "Optimized Parallel"].sort_values("workers")

    workers = sorted(list(set(orig_df["workers"].tolist() + opt_df["workers"].tolist())))
    if not workers:
        ax.text(0.5, 0.5, "No parallel data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    # 100% Ideal Line
    ax.axhline(100.0, color="#64748b", linestyle=":", linewidth=1.5, label="Ideal Efficiency (100%)")

    # Plot Original Efficiency
    if not orig_df.empty:
        w_orig = orig_df["workers"].tolist()
        e_orig = [eff * 100.0 for eff in orig_df["efficiency"]]
        ax.plot(w_orig, e_orig, marker="s", color="#f97316", linewidth=2.0, label="Original Parallel Efficiency")

    # Plot Optimized Efficiency
    if not opt_df.empty:
        w_opt = opt_df["workers"].tolist()
        e_opt = [eff * 100.0 for eff in opt_df["efficiency"]]
        ax.plot(w_opt, e_opt, marker="o", color="#10b981", linewidth=2.2, label="Optimized Parallel Efficiency")

    ax.set_title("Parallel Core Efficiency: Original vs. Optimized", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Worker Process Count", fontsize=10)
    ax.set_ylabel("Parallel Efficiency (%)", fontsize=10)
    ax.set_xticks(workers)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.legend(loc="lower left", fontsize=8.5)

    fig.tight_layout()
    return fig


def plot_throughput_comparison(df: pd.DataFrame) -> plt.Figure:
    """
    Generate bar chart comparing processing throughput in Images/sec and Megapixels/sec.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)

    modes = []
    tp_imgs = []
    tp_mps = []
    colors = []

    seq_r = df[df["mode"] == "Sequential"]
    if not seq_r.empty:
        modes.append("Sequential")
        tp_imgs.append(float(seq_r["throughput_images_s"].iloc[0]))
        tp_mps.append(float(seq_r["throughput_mp_s"].iloc[0]))
        colors.append("#3b82f6")

    orig_r = df[df["mode"] == "Original Parallel"].sort_values("throughput_images_s", ascending=False)
    if not orig_r.empty:
        w = int(orig_r["workers"].iloc[0])
        modes.append(f"Original ({w}w)")
        tp_imgs.append(float(orig_r["throughput_images_s"].iloc[0]))
        tp_mps.append(float(orig_r["throughput_mp_s"].iloc[0]))
        colors.append("#f97316")

    opt_r = df[df["mode"] == "Optimized Parallel"].sort_values("throughput_images_s", ascending=False)
    if not opt_r.empty:
        w = int(opt_r["workers"].iloc[0])
        modes.append(f"Optimized ({w}w)")
        tp_imgs.append(float(opt_r["throughput_images_s"].iloc[0]))
        tp_mps.append(float(opt_r["throughput_mp_s"].iloc[0]))
        colors.append("#10b981")

    # Plot Images/sec
    b1 = ax1.bar(modes, tp_imgs, color=colors, width=0.55, edgecolor="#1e293b")
    ax1.set_title("Throughput (Images / sec)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Images per Second", fontsize=9)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Plot Megapixels/sec
    b2 = ax2.bar(modes, tp_mps, color=colors, width=0.55, edgecolor="#1e293b")
    ax2.set_title("Throughput (Megapixels / sec)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Megapixels per Second", fontsize=9)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")

    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Peak Pipeline Throughput Comparison", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_scalability_crossover(scalability_df: pd.DataFrame) -> plt.Figure:
    """
    Generate workload scalability curve illustrating execution time vs. workload size
    and highlighting the parallel crossover point where Optimized Parallel beats Sequential.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)

    if scalability_df.empty:
        ax.text(0.5, 0.5, "No scalability data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    df_sorted = scalability_df.sort_values("total_megapixels").copy()
    mp_x = df_sorted["total_megapixels"]

    ax.plot(mp_x, df_sorted["seq_time_s"], marker="o", color="#dc2626", linewidth=2.0, label="Sequential (T1)")
    ax.plot(mp_x, df_sorted["orig_par_time_s"], marker="s", color="#f97316", linewidth=1.8, linestyle="--", label="Original Parallel")
    ax.plot(mp_x, df_sorted["opt_par_time_s"], marker="^", color="#10b981", linewidth=2.4, label="Optimized Parallel (SharedMemory)")

    # Find crossover point where opt_par_time_s < seq_time_s
    crossover_pts = df_sorted[df_sorted["opt_speedup"] > 1.0]
    if not crossover_pts.empty:
        first_c = crossover_pts.iloc[0]
        cx = float(first_c["total_megapixels"])
        cy = float(first_c["opt_par_time_s"])
        ax.scatter([cx], [cy], color="#7c3aed", s=140, zorder=6, edgecolors="#ffffff", linewidth=2)
        ax.annotate(
            f"Parallel Crossover\n({first_c['num_images']} imgs, {cx:.1f} MP)\n{first_c['opt_speedup']:.2f}x Speedup",
            xy=(cx, cy),
            xytext=(cx * 0.75, cy * 1.35),
            arrowprops=dict(facecolor="#7c3aed", shrink=0.08, width=1.5, headwidth=6),
            fontsize=8.5,
            fontweight="bold",
            color="#5b21b6",
        )

    ax.set_title("Workload Scalability & Parallel Performance Crossover Analysis", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Total Workload Size (Megapixels)", fontsize=10)
    ax.set_ylabel("Execution Time (seconds)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    return fig


# Backward-compatibility aliases
plot_three_way_speedup = plot_speedup_comparison
plot_three_way_throughput = plot_throughput_comparison
plot_three_way_efficiency = plot_efficiency_comparison
