"""
Comprehensive 3-Way Benchmarking Engine for HPC Performance Evaluation.

Compares three execution configurations under rigorous, fair experimental controls:
1. Sequential Baseline (Single-core, unpickled pure-NumPy execution)
2. Original Multiprocessing (Standard multiprocessing.Pool, full array pickling over IPC)
3. Optimized Multiprocessing (Zero-copy SharedMemory, single-threaded BLAS, adaptive chunking)

Supports both:
- End-to-End Mode: Captures full process pool creation, IPC transfer, compute, and teardown.
- Steady-State Mode: Reuses initialized worker processes; isolates computation throughput from startup overhead.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.metrics import calculate_efficiency, calculate_speedup
from src.parallel import process_images_parallel
from src.parallel_optimized import (
    PersistentWorkerPool,
    process_images_parallel_optimized,
)
from src.pipeline import process_single_image
from src.sequential import process_images_sequential


def run_three_way_benchmark(
    images: List[Tuple[str, np.ndarray]],
    params: Dict[str, Any],
    worker_counts: List[int],
    repeats: int = 3,
    benchmark_mode: str = "steady_state",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute a fair 3-way comparative benchmark across Sequential, Original, and Optimized parallel pipelines.

    Args:
        images: List of (name, rgb_array) pairs.
        params: Filter configuration dictionary ('kernel_size', 'sigma').
        worker_counts: List of core counts to test (e.g. [2, 4, 6, 8, 10]).
        repeats: Repetitions per configuration trial.
        benchmark_mode: 'steady_state' (reused pool) or 'end_to_end' (pool creation timed).
        progress_callback: Optional callback func(steps_done, total_steps, message).
        mode: Optional alias for benchmark_mode.

    Returns:
        Tuple containing:
            - df (pd.DataFrame): Unified benchmark comparison table.
            - last_results (List[Dict[str, Any]]): Processed outputs from final run.
            - metadata (Dict[str, Any]): Benchmark configuration and timing metadata.
    """
    if mode is not None:
        benchmark_mode = mode
    if not images:
        raise ValueError("The images list cannot be empty.")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}.")

    max_cpus = os.cpu_count() or 1
    unique_workers = sorted(list(set(worker_counts)))
    for w in unique_workers:
        if not isinstance(w, int) or w < 1 or w > max_cpus:
            raise ValueError(
                f"Worker count must be an integer between 1 and {max_cpus}, got {w}."
            )

    num_images = len(images)
    total_pixels = sum(arr.shape[0] * arr.shape[1] for _, arr in images)
    total_megapixels = round(total_pixels / 1_000_000.0, 4)

    # Total benchmark steps: Sequential repeats + (len(workers) * repeats * 2 modes)
    total_steps = repeats + (len(unique_workers) * repeats * 2)
    steps_done = 0

    rows: List[Dict[str, Any]] = []
    last_results: List[Dict[str, Any]] = []
    pool_init_times: Dict[int, float] = {}

    # =========================================================================
    # 1. Warm-up Pass for Sequential Pipeline
    # =========================================================================
    warmup_task = (images[0][0], images[0][1], params)
    process_single_image(warmup_task)

    # =========================================================================
    # 2. Benchmark Sequential Baseline (T1)
    # =========================================================================
    seq_times: List[float] = []
    for r in range(repeats):
        res, elapsed = process_images_sequential(images, params)
        seq_times.append(elapsed)
        last_results = res
        steps_done += 1
        if progress_callback:
            progress_callback(
                steps_done,
                total_steps,
                f"Sequential baseline trial {r + 1}/{repeats} ({elapsed:.4f}s)",
            )

    seq_mean = float(np.mean(seq_times))
    seq_median = float(np.median(seq_times))
    seq_min = float(np.min(seq_times))
    seq_std = float(np.std(seq_times, ddof=1)) if repeats > 1 else 0.0
    seq_cv = (seq_std / seq_mean * 100.0) if seq_mean > 0 else 0.0
    seq_tp_img = round(num_images / seq_mean, 2) if seq_mean > 0 else 0.0
    seq_tp_mp = round(total_megapixels / seq_mean, 2) if seq_mean > 0 else 0.0

    rows.append(
        {
            "mode": "Sequential",
            "workers": 1,
            "runs": repeats,
            "mean_time_s": seq_mean,
            "median_time_s": seq_median,
            "min_time_s": seq_min,
            "std_time_s": seq_std,
            "cv_percent": round(seq_cv, 2),
            "speedup": 1.0,
            "efficiency": 1.0,
            "throughput_images_s": seq_tp_img,
            "throughput_mp_s": seq_tp_mp,
            "optimization_gain": 1.0,
            "benchmark_mode": benchmark_mode,
            "num_images": num_images,
            "total_megapixels": total_megapixels,
        }
    )

    # Map of original parallel times per worker count to calculate optimization gain
    orig_parallel_means: Dict[int, float] = {}

    # =========================================================================
    # 3. Benchmark Original Parallel Multiprocessing
    # =========================================================================
    for workers in unique_workers:
        par_times: List[float] = []
        for r in range(repeats):
            par_res, elapsed = process_images_parallel(images, params, num_workers=workers)
            par_times.append(elapsed)
            last_results = par_res
            steps_done += 1
            if progress_callback:
                progress_callback(
                    steps_done,
                    total_steps,
                    f"Original Parallel ({workers}w) trial {r + 1}/{repeats} ({elapsed:.4f}s)",
                )

        mean_t = float(np.mean(par_times))
        median_t = float(np.median(par_times))
        min_t = float(np.min(par_times))
        std_t = float(np.std(par_times, ddof=1)) if repeats > 1 else 0.0
        cv_pct = (std_t / mean_t * 100.0) if mean_t > 0 else 0.0
        sp = calculate_speedup(seq_mean, mean_t)
        eff = calculate_efficiency(sp, workers)
        tp_img = round(num_images / mean_t, 2) if mean_t > 0 else 0.0
        tp_mp = round(total_megapixels / mean_t, 2) if mean_t > 0 else 0.0

        orig_parallel_means[workers] = mean_t

        rows.append(
            {
                "mode": "Original Parallel",
                "workers": workers,
                "runs": repeats,
                "mean_time_s": mean_t,
                "median_time_s": median_t,
                "min_time_s": min_t,
                "std_time_s": std_t,
                "cv_percent": round(cv_pct, 2),
                "speedup": sp,
                "efficiency": eff,
                "throughput_images_s": tp_img,
                "throughput_mp_s": tp_mp,
                "optimization_gain": 1.0,  # Baseline for original
                "benchmark_mode": benchmark_mode,
                "num_images": num_images,
                "total_megapixels": total_megapixels,
            }
        )

    # =========================================================================
    # 4. Benchmark Optimized Parallel (SharedMemory & Single-Threaded BLAS)
    # =========================================================================
    for workers in unique_workers:
        opt_times: List[float] = []

        if benchmark_mode == "steady_state":
            persistent_pool = PersistentWorkerPool(num_workers=workers)
            init_time = persistent_pool.start()
            pool_init_times[workers] = init_time

            # Pre-warm pool with untimed single task
            try:
                process_images_parallel_optimized(
                    images[:1],
                    params,
                    num_workers=workers,
                    persistent_pool=persistent_pool,
                )
            except Exception:
                pass

            try:
                for r in range(repeats):
                    opt_res, elapsed = process_images_parallel_optimized(
                        images,
                        params,
                        num_workers=workers,
                        persistent_pool=persistent_pool,
                    )
                    opt_times.append(elapsed)
                    last_results = opt_res
                    steps_done += 1
                    if progress_callback:
                        progress_callback(
                            steps_done,
                            total_steps,
                            f"Optimized Parallel ({workers}w steady) trial {r + 1}/{repeats} ({elapsed:.4f}s)",
                        )
            finally:
                persistent_pool.shutdown()

        else:
            # End-to-End mode: Pool startup and teardown are measured per trial
            for r in range(repeats):
                opt_res, elapsed = process_images_parallel_optimized(
                    images,
                    params,
                    num_workers=workers,
                    persistent_pool=None,
                )
                opt_times.append(elapsed)
                last_results = opt_res
                steps_done += 1
                if progress_callback:
                    progress_callback(
                        steps_done,
                        total_steps,
                        f"Optimized Parallel ({workers}w e2e) trial {r + 1}/{repeats} ({elapsed:.4f}s)",
                    )

        mean_opt = float(np.mean(opt_times))
        median_opt = float(np.median(opt_times))
        min_opt = float(np.min(opt_times))
        std_opt = float(np.std(opt_times, ddof=1)) if repeats > 1 else 0.0
        cv_opt = (std_opt / mean_opt * 100.0) if mean_opt > 0 else 0.0
        sp_opt = calculate_speedup(seq_mean, mean_opt)
        eff_opt = calculate_efficiency(sp_opt, workers)
        tp_img_opt = round(num_images / mean_opt, 2) if mean_opt > 0 else 0.0
        tp_mp_opt = round(total_megapixels / mean_opt, 2) if mean_opt > 0 else 0.0

        # Optimization gain: how much faster Optimized is vs Original for the same worker count
        orig_t = orig_parallel_means.get(workers, mean_opt)
        gain = round(orig_t / mean_opt, 2) if mean_opt > 0 else 1.0

        rows.append(
            {
                "mode": "Optimized Parallel",
                "workers": workers,
                "runs": repeats,
                "mean_time_s": mean_opt,
                "median_time_s": median_opt,
                "min_time_s": min_opt,
                "std_time_s": std_opt,
                "cv_percent": round(cv_opt, 2),
                "speedup": sp_opt,
                "efficiency": eff_opt,
                "throughput_images_s": tp_img_opt,
                "throughput_mp_s": tp_mp_opt,
                "optimization_gain": gain,
                "benchmark_mode": benchmark_mode,
                "num_images": num_images,
                "total_megapixels": total_megapixels,
            }
        )

    columns = [
        "mode",
        "workers",
        "runs",
        "mean_time_s",
        "median_time_s",
        "min_time_s",
        "std_time_s",
        "cv_percent",
        "speedup",
        "efficiency",
        "throughput_images_s",
        "throughput_mp_s",
        "optimization_gain",
        "benchmark_mode",
        "num_images",
        "total_megapixels",
    ]
    df = pd.DataFrame(rows, columns=columns)

    metadata = {
        "benchmark_mode": benchmark_mode,
        "repeats": repeats,
        "num_images": num_images,
        "total_megapixels": total_megapixels,
        "pool_init_times_s": pool_init_times,
        "tested_workers": unique_workers,
    }

    return df, last_results, metadata
