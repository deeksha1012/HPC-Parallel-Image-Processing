"""
Benchmarking engine for comparing sequential and parallel image processing.

Measures wall-clock execution times across multiple worker process counts,
computes speedup and parallel efficiency, records system specifications,
and outputs structured data to CSV and JSON for downstream visualization.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_KERNEL_SIZE, DEFAULT_REPEATS, DEFAULT_SIGMA, RESULTS_DIR, SAMPLE_IMAGES_DIR
from src.image_io import load_images_from_folder
from src.metrics import calculate_efficiency, calculate_speedup
from src.parallel import process_images_parallel
from src.pipeline import process_single_image
from src.sequential import process_images_sequential


def get_system_info() -> dict[str, Any]:
    """
    Collect hardware and software runtime environment details.

    Returns:
        dict[str, Any]: System metadata dictionary containing:
            - 'timestamp': ISO formatted timestamp of the benchmark run.
            - 'platform': Operating system release and kernel details.
            - 'machine': Hardware architecture (e.g., 'arm64', 'x86_64').
            - 'python_version': Active Python runtime version string.
            - 'cpu_count': Total logical CPU cores reported by the OS.
            - 'start_method': Multiprocessing start method ('spawn').
    """
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
        "start_method": "spawn",
    }


def run_benchmark(
    images: list[tuple[str, np.ndarray]],
    params: dict[str, Any],
    worker_counts: list[int],
    repeats: int = 3,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Execute a fair, repeatable benchmark comparing sequential vs. parallel processing.

    Protocol:
    1. Validate input images, worker counts, and repeat parameters.
    2. Perform one untimed warm-up pass on the sequential pipeline to remove
       first-call Python/OS initialization overhead from the baseline.
    3. Measure sequential execution across 'repeats' runs and average them.
    4. Measure parallel execution across 'repeats' runs for each worker count.
       (Parallel runs are deliberately not warmed up to include pool startup costs).
    5. Calculate speedup and efficiency using the mean sequential duration as baseline.

    Args:
        images: List of (filename, rgb_uint8_array) pairs.
        params: Filter configuration dictionary ('kernel_size', 'sigma').
        worker_counts: List of positive integer core counts to test.
        repeats: Number of trials per configuration (default: 3).
        progress_callback: Optional callback func(steps_done, total_steps, message).

    Returns:
        tuple containing:
            - df (pd.DataFrame): Benchmark summary table with timings and metrics.
            - last_results (list[dict]): Processed image results from the last sequential run.

    Raises:
        ValueError: If images is empty, repeats < 1, or any worker count is invalid.
    """
    # 1. Input validations
    if not images:
        raise ValueError("The images list cannot be empty.")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}.")

    max_cpus = os.cpu_count() or 1
    # De-duplicate and sort worker counts in ascending order
    unique_workers = sorted(list(set(worker_counts)))
    for w in unique_workers:
        if not isinstance(w, int) or w < 1 or w > max_cpus:
            raise ValueError(
                f"Worker count must be an integer between 1 and {max_cpus}, got {w}."
            )

    # Calculate total processing workload in megapixels
    total_pixels = sum(arr.shape[0] * arr.shape[1] for _, arr in images)
    total_megapixels = round(total_pixels / 1_000_000.0, 4)

    # 2. Untimed warm-up call on sequential pipeline
    # Ensures library initialization and memory allocation don't penalize sequential T1
    warmup_task = (images[0][0], images[0][1], params)
    process_single_image(warmup_task)

    # Compute total steps for progress reporting: sequential + (len(workers) * repeats)
    total_steps = repeats + len(unique_workers) * repeats
    steps_done = 0

    # 3. Benchmark Sequential execution (Baseline)
    seq_times: list[float] = []
    last_results: list[dict[str, Any]] = []

    for r in range(repeats):
        results, elapsed = process_images_sequential(images, params)
        seq_times.append(elapsed)
        last_results = results
        steps_done += 1
        if progress_callback is not None:
            progress_callback(
                steps_done,
                total_steps,
                f"Sequential run {r + 1}/{repeats} ({elapsed:.4f}s)",
            )

    seq_mean = float(np.mean(seq_times))
    seq_min = float(np.min(seq_times))
    seq_std = float(np.std(seq_times, ddof=1)) if repeats > 1 else 0.0
    seq_speedup = calculate_speedup(seq_mean, seq_mean)
    seq_efficiency = calculate_efficiency(seq_speedup, 1)
    seq_times_str = ";".join(f"{t:.4f}" for t in seq_times)

    rows: list[dict[str, Any]] = [
        {
            "mode": "Sequential",
            "workers": 1,
            "runs": repeats,
            "mean_time_s": seq_mean,
            "min_time_s": seq_min,
            "std_time_s": seq_std,
            "speedup": seq_speedup,
            "efficiency": seq_efficiency,
            "num_images": len(images),
            "total_megapixels": total_megapixels,
            "run_times_s": seq_times_str,
        }
    ]

    # 4. Benchmark Parallel execution across requested worker counts
    for workers in unique_workers:
        par_times: list[float] = []
        for r in range(repeats):
            _, elapsed = process_images_parallel(
                images=images,
                params=params,
                num_workers=workers,
            )
            par_times.append(elapsed)
            steps_done += 1
            if progress_callback is not None:
                progress_callback(
                    steps_done,
                    total_steps,
                    f"Parallel ({workers} workers) run {r + 1}/{repeats} ({elapsed:.4f}s)",
                )

        par_mean = float(np.mean(par_times))
        par_min = float(np.min(par_times))
        par_std = float(np.std(par_times, ddof=1)) if repeats > 1 else 0.0
        speedup = calculate_speedup(seq_mean, par_mean)
        efficiency = calculate_efficiency(speedup, workers)
        par_times_str = ";".join(f"{t:.4f}" for t in par_times)

        rows.append(
            {
                "mode": "Parallel",
                "workers": workers,
                "runs": repeats,
                "mean_time_s": par_mean,
                "min_time_s": par_min,
                "std_time_s": par_std,
                "speedup": speedup,
                "efficiency": efficiency,
                "num_images": len(images),
                "total_megapixels": total_megapixels,
                "run_times_s": par_times_str,
            }
        )

    columns = [
        "mode",
        "workers",
        "runs",
        "mean_time_s",
        "min_time_s",
        "std_time_s",
        "speedup",
        "efficiency",
        "num_images",
        "total_megapixels",
        "run_times_s",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df, last_results


def main():
    """Command-line interface for running benchmarks and exporting results."""
    cpu_limit = os.cpu_count() or 1
    # Default workers: 2, 4, and max CPU count, de-duplicated and bounded
    default_workers = sorted(list(set([w for w in [2, 4, cpu_limit] if 1 <= w <= cpu_limit])))

    parser = argparse.ArgumentParser(
        description="Run automated HPC image processing benchmarks."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SAMPLE_IMAGES_DIR),
        help="Input folder containing images (default: sample_images)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=default_workers,
        help=f"Worker counts to benchmark (default: {default_workers})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repetitions per test (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        help=f"Gaussian blur kernel size (default: {DEFAULT_KERNEL_SIZE})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Gaussian blur sigma (default: {DEFAULT_SIGMA})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "benchmark_results.csv"),
        help="Output CSV file path (default: results/benchmark_results.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    images = load_images_from_folder(input_path)

    params = {
        "kernel_size": args.kernel_size,
        "sigma": args.sigma,
    }

    # 1. Collect and save system specifications
    sys_info = get_system_info()
    out_csv = Path(args.output)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sys_info_path = out_dir / "system_info.json"
    with open(sys_info_path, "w", encoding="utf-8") as f:
        json.dump(sys_info, f, indent=2)

    # 2. Print system information
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Timestamp:      {sys_info['timestamp']}")
    print(f"Platform:       {sys_info['platform']}")
    print(f"Machine:        {sys_info['machine']}")
    print(f"Python Version: {sys_info['python_version']}")
    print(f"CPU Count:      {sys_info['cpu_count']}")
    print(f"Start Method:   {sys_info['start_method']}")
    print("=" * 60)
    print(f"Running benchmark on {len(images)} image(s) with {args.repeats} repeat(s)...")
    print(f"Worker configurations: {sorted(list(set(args.workers)))}\n")

    # 3. Run the benchmark suite
    df, _ = run_benchmark(
        images=images,
        params=params,
        worker_counts=args.workers,
        repeats=args.repeats,
    )

    # 4. Save full precision DataFrame to CSV
    df.to_csv(out_csv, index=False)
    print(f"Benchmark results saved to: {out_csv}")
    print(f"System specifications saved to: {sys_info_path}\n")

    # 5. Format DataFrame rounded to 4 decimals for display only
    display_df = df.copy()
    float_cols = ["mean_time_s", "min_time_s", "std_time_s", "speedup", "efficiency"]
    for col in float_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    print("BENCHMARK RESULTS SUMMARY:")
    print(display_df.to_string(index=False))
    print("\nTimes measured on this machine; they will vary between runs.")


if __name__ == "__main__":
    main()
