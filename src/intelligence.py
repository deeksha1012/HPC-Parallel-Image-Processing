"""
Performance Intelligence and Automated HPC Analytical Reasoning Engine.

Performs rule-based evaluation of benchmark scaling, resource consumption,
and architectural bottlenecks without external AI APIs or cloud dependencies.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_throughput_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment benchmark DataFrame with HPC throughput and variability metrics:
    - throughput_images_per_s: Images processed per second (N / T).
    - throughput_mp_per_s: Megapixels processed per second (MP / T).
    - cv_percent: Coefficient of Variation (Std Dev / Mean * 100), measuring runtime stability.
    - ideal_speedup: Theoretical linear speedup equal to worker count.
    - speedup_deficit: Gap between ideal speedup and measured speedup (ideal - actual).

    Args:
        df: Benchmark DataFrame.

    Returns:
        pd.DataFrame: Augmented DataFrame with additional analytical columns.
    """
    df_aug = df.copy()

    # Images/sec throughput
    df_aug["throughput_images_per_s"] = df_aug.apply(
        lambda r: round(float(r["num_images"]) / float(r["mean_time_s"]), 2)
        if float(r["mean_time_s"]) > 0 else 0.0,
        axis=1,
    )

    # Megapixels/sec throughput
    df_aug["throughput_mp_per_s"] = df_aug.apply(
        lambda r: round(float(r["total_megapixels"]) / float(r["mean_time_s"]), 2)
        if float(r["mean_time_s"]) > 0 else 0.0,
        axis=1,
    )

    # Coefficient of Variation (Execution Time Stability)
    df_aug["cv_percent"] = df_aug.apply(
        lambda r: round((float(r["std_time_s"]) / float(r["mean_time_s"])) * 100.0, 2)
        if float(r["mean_time_s"]) > 0 else 0.0,
        axis=1,
    )

    # Ideal linear speedup and deficit
    df_aug["ideal_speedup"] = df_aug["workers"].astype(float)
    df_aug["speedup_deficit"] = df_aug.apply(
        lambda r: round(max(0.0, float(r["ideal_speedup"]) - float(r["speedup"])), 2),
        axis=1,
    )

    return df_aug


def analyze_performance(
    df: pd.DataFrame,
    monitor_summary: Optional[Dict[str, Any]] = None,
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute rule-based performance intelligence analysis across benchmark results.

    Evaluates speedup curves, core efficiency degradation, IPC overhead,
    Apple Silicon core asymmetry, and system resource telemetry to produce
    actionable HPC recommendations.

    Args:
        df: Benchmark DataFrame with timing, speedup, and efficiency metrics.
        monitor_summary: Optional dictionary with background CPU and RAM metrics.
        hardware_info: Optional dictionary with CPU and platform specifications.

    Returns:
        Dict[str, Any]: Structured performance intelligence findings containing:
            - 'fastest_overall': Dict with mode, workers, time_s
            - 'fastest_parallel': Dict with workers, time_s, speedup, efficiency
            - 'sequential_baseline': Dict with time_s, throughput_mp_per_s
            - 'recommended_workers': Optimal worker process count
            - 'recommendation_reason': Clear architectural justification
            - 'speedup_trend': Classification ('linear', 'sublinear', 'diminishing', 'negative')
            - 'efficiency_status': Efficiency assessment
            - 'resource_observations': Observations on CPU and RAM telemetry
            - 'key_takeaways': Bullet points summarizing findings
    """
    if df.empty:
        return {
            "fastest_overall": {"mode": "None", "workers": 0, "time_s": 0.0},
            "fastest_parallel": None,
            "sequential_baseline": {"time_s": 0.0, "throughput_mp_per_s": 0.0},
            "recommended_workers": 1,
            "recommendation_reason": "No benchmark data available to evaluate.",
            "speedup_trend": "none",
            "efficiency_status": "none",
            "resource_observations": "No telemetry recorded.",
            "key_takeaways": [],
        }

    df_aug = compute_throughput_metrics(df)

    seq_row = df_aug[df_aug["mode"] == "Sequential"].iloc[0]
    seq_time = float(seq_row["mean_time_s"])
    seq_mp_s = float(seq_row["throughput_mp_per_s"])

    par_df = df_aug[df_aug["mode"] == "Parallel"].copy()

    # 1. Fastest overall run
    fastest_overall_idx = df_aug["mean_time_s"].idxmin()
    fastest_row = df_aug.loc[fastest_overall_idx]
    fastest_overall = {
        "mode": str(fastest_row["mode"]),
        "workers": int(fastest_row["workers"]),
        "time_s": round(float(fastest_row["mean_time_s"]), 4),
        "speedup": round(float(fastest_row["speedup"]), 2),
        "efficiency_pct": round(float(fastest_row["efficiency"]) * 100.0, 1),
        "throughput_mp_s": round(float(fastest_row["throughput_mp_per_s"]), 2),
    }

    # 2. Sequential baseline details
    seq_baseline = {
        "time_s": round(seq_time, 4),
        "throughput_mp_s": round(seq_mp_s, 2),
    }

    takeaways: List[str] = []

    # Case A: Only Sequential run performed
    if par_df.empty:
        return {
            "fastest_overall": fastest_overall,
            "fastest_parallel": None,
            "sequential_baseline": seq_baseline,
            "recommended_workers": 1,
            "recommendation_reason": "Only sequential execution was evaluated. Select parallel worker counts to assess multi-core scaling.",
            "speedup_trend": "sequential_only",
            "efficiency_status": "100.0% (Single Core)",
            "resource_observations": "Single-core sequential execution utilized 1 process.",
            "key_takeaways": ["Sequential baseline established.", "Evaluate parallel workers to compare speedup."],
        }

    # 3. Fastest parallel configuration
    par_min_idx = par_df["mean_time_s"].idxmin()
    fastest_par_row = par_df.loc[par_min_idx]
    fastest_par_workers = int(fastest_par_row["workers"])
    fastest_par_time = float(fastest_par_row["mean_time_s"])
    fastest_par_speedup = float(fastest_par_row["speedup"])
    fastest_par_eff = float(fastest_par_row["efficiency"]) * 100.0

    fastest_parallel = {
        "workers": fastest_par_workers,
        "time_s": round(fastest_par_time, 4),
        "speedup": round(fastest_par_speedup, 2),
        "efficiency_pct": round(fastest_par_eff, 1),
        "throughput_mp_s": round(float(fastest_par_row["throughput_mp_per_s"]), 2),
    }

    # Percent improvement or slowdown
    if fastest_par_time < seq_time:
        time_diff_pct = round(((seq_time - fastest_par_time) / seq_time) * 100.0, 1)
        speedup_summary = f"{fastest_par_speedup:.2f}x faster ({time_diff_pct}% reduction in runtime)"
    else:
        time_diff_pct = round(((fastest_par_time - seq_time) / seq_time) * 100.0, 1)
        speedup_summary = f"{time_diff_pct}% slower than sequential baseline due to IPC overhead"

    # 4. Analyze Scaling Trend and Efficiency Degradation
    par_sorted = par_df.sort_values("workers")
    min_eff_row = par_df.loc[par_df["efficiency"].idxmin()]
    min_eff_val = float(min_eff_row["efficiency"]) * 100.0
    min_eff_workers = int(min_eff_row["workers"])

    if fastest_par_speedup < 1.0:
        speedup_trend = "negative"
        efficiency_status = "Severely degraded by process spawn and serialization overhead"
    elif fastest_par_eff >= 80.0:
        speedup_trend = "near_linear"
        efficiency_status = f"Excellent scaling (≥80% efficiency maintained up to {fastest_par_workers} workers)"
    elif fastest_par_eff >= 50.0:
        speedup_trend = "sublinear"
        efficiency_status = f"Sublinear scaling (efficiency drops to {min_eff_val:.1f}% at {min_eff_workers} workers)"
    else:
        speedup_trend = "diminishing"
        efficiency_status = f"Diminishing returns (efficiency falls to {min_eff_val:.1f}% at {min_eff_workers} workers)"

    # 5. Recommendation Logic: Find the optimal balance between speedup and efficiency
    # Goal: pick worker count with highest speedup unless a lower core count achieves
    # >= 90% of that speedup with significantly higher efficiency (>15% higher).
    candidate_workers = fastest_par_workers
    candidate_reason = ""

    # Sort parallel workers ascending
    for _, row in par_sorted.iterrows():
        w = int(row["workers"])
        sp = float(row["speedup"])
        eff = float(row["efficiency"]) * 100.0

        if w < fastest_par_workers and sp >= (0.90 * fastest_par_speedup) and (eff >= fastest_par_eff + 15.0):
            candidate_workers = w
            candidate_reason = (
                f"Selected {w} workers as the optimal sweet spot: it delivers {sp:.2f}x speedup "
                f"({(sp / fastest_par_speedup) * 100:.1f}% of peak performance) while maintaining "
                f"significantly higher parallel efficiency ({eff:.1f}% vs {fastest_par_eff:.1f}% at {fastest_par_workers} workers). "
                f"This avoids core saturation and diminishing IPC returns."
            )
            break

    if not candidate_reason:
        if fastest_par_time < seq_time:
            candidate_workers = fastest_par_workers
            candidate_reason = (
                f"Selected {fastest_par_workers} workers: achieves minimum wall-clock execution time "
                f"({fastest_par_time:.3f} s) and peak speedup of {fastest_par_speedup:.2f}x "
                f"with {fastest_par_eff:.1f}% parallel efficiency."
            )
        else:
            candidate_workers = 1
            candidate_reason = (
                f"Selected Sequential (1 worker): For this workload size ({seq_row['num_images']} images, "
                f"{seq_row['total_megapixels']} MP), the overhead of spawning worker processes and pickling "
                f"arrays across inter-process communication (IPC) pipes outweighs parallel compute gains. "
                f"Sequential execution is {seq_time:.3f}s vs {fastest_par_time:.3f}s parallel."
            )

    # 6. Apple Silicon Architecture Check (P-cores vs E-cores)
    physical_cores = 10
    if hardware_info:
        physical_cores = int(hardware_info.get("physical_cores", 10))

    apple_silicon_note = ""
    is_apple_silicon = False
    if hardware_info:
        mach = hardware_info.get("machine", "").lower()
        plat = hardware_info.get("platform", "").lower()
        proc = hardware_info.get("processor", "").lower()
        if "arm" in mach or "apple" in proc or "darwin" in plat:
            is_apple_silicon = True

    if is_apple_silicon:
        high_workers = [w for w in par_df["workers"].astype(int) if w > (physical_cores // 2)]
        if high_workers:
            apple_silicon_note = (
                f"Apple Silicon dynamically schedules threads across asymmetric Performance (P-cores) "
                f"and Efficiency (E-cores). Worker counts exceeding {physical_cores // 2} engage lower-clocked "
                f"E-cores, causing slight load imbalance where faster cores wait for slower cores."
            )

    # 7. Resource Observations from Monitor Summary
    resource_obs_list = []
    if monitor_summary:
        avg_cpu = monitor_summary.get("avg_cpu_percent", 0.0)
        peak_cpu = monitor_summary.get("peak_cpu_percent", 0.0)
        peak_ram_mb = monitor_summary.get("peak_memory_mb", 0.0)
        peak_ram_pct = monitor_summary.get("peak_memory_percent", 0.0)
        growth_mb = monitor_summary.get("memory_growth_mb", 0.0)
        high_mem = monitor_summary.get("high_memory_warning", False)

        resource_obs_list.append(
            f"Average CPU utilization was {avg_cpu:.1f}% (peak: {peak_cpu:.1f}%)."
        )
        resource_obs_list.append(
            f"Peak memory footprint reached {peak_ram_mb:.1f} MB ({peak_ram_pct:.1f}% of total RAM), "
            f"with {growth_mb:.1f} MB net growth during processing."
        )
        if high_mem:
            resource_obs_list.append(
                "⚠️ High memory utilization detected (>85%). Consider reducing batch sizes to avoid paging."
            )
        else:
            resource_obs_list.append(
                "System memory remained well within safe operational limits throughout benchmark execution."
            )
    else:
        resource_obs_list.append("Background telemetry was not enabled during this benchmark.")

    resource_observations = " ".join(resource_obs_list)

    # 8. Assemble Key Takeaways
    takeaways.append(
        f"<b>Fastest Overall:</b> {fastest_overall['mode']} ({fastest_overall['workers']}w) at {fastest_overall['time_s']:.3f} s."
    )
    if fastest_par_time < seq_time:
        takeaways.append(
            f"<b>Speedup Achieved:</b> {fastest_par_speedup:.2f}x speedup over sequential baseline ({fastest_par_workers} workers)."
        )
    else:
        takeaways.append(
            "<b>Overhead Dominance:</b> Multiprocessing spawn and IPC overhead exceeded parallel speedup on this dataset."
        )
    takeaways.append(f"<b>Core Scaling:</b> Parallel efficiency reached a minimum of {min_eff_val:.1f}% at {min_eff_workers} workers.")
    takeaways.append(f"<b>Recommended Allocation:</b> {candidate_workers} worker processes ({candidate_reason.split(':')[0]}).")

    if apple_silicon_note:
        takeaways.append(f"<b>Architecture:</b> {apple_silicon_note}")

    return {
        "fastest_overall": fastest_overall,
        "fastest_parallel": fastest_parallel,
        "sequential_baseline": seq_baseline,
        "recommended_workers": candidate_workers,
        "recommendation_reason": candidate_reason,
        "speedup_summary": speedup_summary,
        "speedup_trend": speedup_trend,
        "efficiency_status": efficiency_status,
        "resource_observations": resource_observations,
        "apple_silicon_note": apple_silicon_note,
        "key_takeaways": takeaways,
        "augmented_df": df_aug,
    }
