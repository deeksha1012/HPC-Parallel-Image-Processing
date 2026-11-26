"""
Workload Scalability and Parallel Crossover Detection Engine.

Provides:
1. Automated memory safety estimation prior to large image allocations.
2. Scalability experiment runner evaluating scaling across image counts, resolutions, and kernels.
3. Analytical crossover detection identifying the exact workload threshold where parallel
   acceleration overcomes process management and IPC latency.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import psutil

from src.parallel import process_images_parallel
from src.parallel_optimized import (
    PersistentWorkerPool,
    process_images_parallel_optimized,
)
from src.sample_generator import generate_sample_images_in_memory
from src.sequential import process_images_sequential


def estimate_memory_requirements(
    count: int,
    width: int,
    height: int,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Estimate RAM consumption for a candidate workload and check system safety.

    Args:
        count: Number of images in the batch.
        width: Image width in pixels.
        height: Image height in pixels.
        num_workers: Number of parallel worker processes.

    Returns:
        Dict[str, Any]: Memory breakdown and safety assessment.
    """
    bytes_per_image_rgb = width * height * 3
    bytes_per_image_stage = width * height  # 1 channel uint8 for gray, blur, edges
    total_stages_bytes = bytes_per_image_stage * 3

    input_mb = (count * bytes_per_image_rgb) / (1024.0 * 1024.0)
    stages_mb = (count * total_stages_bytes) / (1024.0 * 1024.0)
    shm_total_mb = input_mb + stages_mb

    # Working memory per worker for float32 convolution buffers (~2-3 padded planes)
    worker_working_mb = (num_workers * width * height * 4 * 3) / (1024.0 * 1024.0)
    estimated_peak_mb = shm_total_mb + worker_working_mb

    vm = psutil.virtual_memory()
    available_mb = vm.available / (1024.0 * 1024.0)
    total_ram_mb = vm.total / (1024.0 * 1024.0)

    # Safe threshold: workload peak should not exceed 65% of available memory
    is_safe = estimated_peak_mb <= (0.65 * available_mb)
    is_critical = estimated_peak_mb >= (0.90 * available_mb)

    warning_msg = ""
    if is_critical:
        warning_msg = (
            f"Workload requires ~{estimated_peak_mb:.0f} MB, which exceeds 90% of available RAM "
            f"({available_mb:.0f} MB). High risk of OS swapping/paging or OOM termination."
        )
    elif not is_safe:
        warning_msg = (
            f"Workload requires ~{estimated_peak_mb:.0f} MB, consuming a substantial fraction of available RAM "
            f"({available_mb:.0f} MB). Recommended to reduce count or resolution."
        )

    return {
        "count": count,
        "width": width,
        "height": height,
        "input_mb": round(input_mb, 2),
        "stages_mb": round(stages_mb, 2),
        "shm_total_mb": round(shm_total_mb, 2),
        "estimated_peak_mb": round(estimated_peak_mb, 2),
        "available_mb": round(available_mb, 2),
        "total_ram_mb": round(total_ram_mb, 2),
        "is_safe": is_safe,
        "is_critical": is_critical,
        "warning_message": warning_msg,
    }


def find_crossover_point(scalability_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze workload scaling results to identify the parallel performance crossover point.

    The crossover point is defined as the workload size (image count or megapixels)
    where Optimized Parallel execution becomes faster than Sequential execution
    (Speedup > 1.0, wall-clock time is reduced).

    Args:
        scalability_df: DataFrame containing columns:
                        ['workload_label', 'num_images', 'total_megapixels',
                         'seq_time_s', 'orig_par_time_s', 'opt_par_time_s',
                         'opt_speedup', 'opt_gain']

    Returns:
        Dict[str, Any]: Analysis containing:
            - 'crossover_observed': bool
            - 'crossover_images': Optional[int]
            - 'crossover_megapixels': Optional[float]
            - 'crossover_label': Optional[str]
            - 'analysis_text': str
    """
    if scalability_df.empty:
        return {
            "crossover_observed": False,
            "crossover_images": None,
            "crossover_megapixels": None,
            "crossover_label": None,
            "analysis_text": "No scalability data available.",
        }

    df_sorted = scalability_df.sort_values("total_megapixels").copy()

    # Find the first workload where opt_speedup > 1.0 (optimized parallel is faster)
    crossover_row = None
    for _, row in df_sorted.iterrows():
        if float(row["opt_speedup"]) > 1.0:
            crossover_row = row
            break

    if crossover_row is not None:
        c_images = int(crossover_row["num_images"])
        c_mp = float(crossover_row["total_megapixels"])
        c_label = str(crossover_row["workload_label"])
        sp = float(crossover_row["opt_speedup"])

        analysis = (
            f"**Parallel Crossover Point Observed:** At **{c_images} images** ({c_mp:.2f} Megapixels, {c_label}), "
            f"Optimized Parallel execution achieved a speedup of **{sp:.2f}x**, successfully amortizing process dispatch "
            f"and IPC scheduling overhead to outperform the single-core sequential baseline. For smaller workloads, "
            f"fixed Python process coordination costs exceed the computational benefit of multi-core parallelism."
        )
        return {
            "crossover_observed": True,
            "crossover_images": c_images,
            "crossover_megapixels": c_mp,
            "crossover_label": c_label,
            "speedup_at_crossover": sp,
            "analysis_text": analysis,
        }
    else:
        # Check if parallel was always slower or always faster
        all_faster = (df_sorted["opt_speedup"] > 1.0).all()
        if all_faster:
            analysis = (
                "Optimized Parallel processing outperformed sequential execution across all tested workload increments."
            )
        else:
            analysis = (
                "Parallel performance crossover was not reached within the tested workload increments. "
                "Sequential processing remained faster across all evaluated sizes due to lightweight filter operations "
                "relative to process dispatch costs. To observe crossover, evaluate larger resolutions (≥1024px) or larger kernel sizes."
            )
        return {
            "crossover_observed": False,
            "crossover_images": None,
            "crossover_megapixels": None,
            "crossover_label": None,
            "speedup_at_crossover": None,
            "analysis_text": analysis,
        }


def run_scalability_experiment(
    workload_configs: List[Dict[str, Any]],
    params: Dict[str, Any],
    num_workers: int = 4,
    benchmark_mode: str = "steady_state",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run an automated scalability sweep across multiple synthetic workload configurations.

    Args:
        workload_configs: List of dicts with 'label', 'count', 'width', 'height'.
        params: Filter configuration dictionary ('kernel_size', 'sigma').
        num_workers: Fixed worker process count to evaluate.
        benchmark_mode: 'steady_state' or 'end_to_end'.
        progress_callback: Optional callback func(steps_done, total_steps, message).

    Returns:
        Tuple containing:
            - scalability_df (pd.DataFrame)
            - crossover_info (Dict[str, Any])
    """
    if not workload_configs:
        raise ValueError("Workload configs list cannot be empty.")

    total_configs = len(workload_configs)
    rows: List[Dict[str, Any]] = []

    # Use a persistent pool for steady-state if requested
    persistent_pool = None
    if benchmark_mode == "steady_state":
        persistent_pool = PersistentWorkerPool(num_workers=num_workers)
        persistent_pool.start()

    try:
        for idx, cfg in enumerate(workload_configs, start=1):
            label = cfg.get("label", f"Workload {idx}")
            count = int(cfg["count"])
            width = int(cfg["width"])
            height = int(cfg["height"])

            if progress_callback:
                progress_callback(
                    idx,
                    total_configs,
                    f"Testing {label} ({count} images, {width}x{height}px)...",
                )

            # Check memory safety
            mem_info = estimate_memory_requirements(count, width, height, num_workers)
            if mem_info["is_critical"]:
                continue

            # Generate synthetic test images
            images = generate_sample_images_in_memory(
                count=count,
                width=width,
                height=height,
                seed=42 + idx,
            )
            total_mp = round((count * width * height) / 1_000_000.0, 4)

            # 1. Measure Sequential
            _, seq_t = process_images_sequential(images, params)

            # 2. Measure Original Parallel
            _, orig_t = process_images_parallel(images, params, num_workers=num_workers)

            # 3. Measure Optimized Parallel
            _, opt_t = process_images_parallel_optimized(
                images,
                params,
                num_workers=num_workers,
                persistent_pool=persistent_pool if benchmark_mode == "steady_state" else None,
            )

            opt_sp = round(seq_t / opt_t, 2) if opt_t > 0 else 0.0
            orig_sp = round(seq_t / orig_t, 2) if orig_t > 0 else 0.0
            opt_gain = round(orig_t / opt_t, 2) if opt_t > 0 else 0.0

            seq_tp_img = round(count / seq_t, 2) if seq_t > 0 else 0.0
            opt_tp_img = round(count / opt_t, 2) if opt_t > 0 else 0.0

            rows.append(
                {
                    "workload_label": label,
                    "num_images": count,
                    "resolution": f"{width}x{height}",
                    "total_megapixels": total_mp,
                    "workers": num_workers,
                    "seq_time_s": round(seq_t, 4),
                    "orig_par_time_s": round(orig_t, 4),
                    "opt_par_time_s": round(opt_t, 4),
                    "orig_speedup": orig_sp,
                    "opt_speedup": opt_sp,
                    "opt_gain": opt_gain,
                    "seq_throughput_images_s": seq_tp_img,
                    "opt_throughput_images_s": opt_tp_img,
                }
            )

    finally:
        if persistent_pool is not None:
            persistent_pool.shutdown()

    df = pd.DataFrame(rows)
    crossover_info = find_crossover_point(df)
    return df, crossover_info
