"""
Automated unit and integration tests for Module 11:
Parallel Performance Optimization & Workload Scalability Suite.
"""

from multiprocessing.shared_memory import SharedMemory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.benchmark_optimized import run_three_way_benchmark
from src.parallel_optimized import (
    PersistentWorkerPool,
    process_images_parallel_optimized,
    shared_memory_batch_scope,
)
from src.pdf_report import generate_benchmark_pdf
from src.plots_optimized import (
    plot_efficiency_comparison,
    plot_scalability_crossover,
    plot_speedup_comparison,
    plot_three_way_execution_time,
    plot_throughput_comparison,
)
from src.scalability import (
    estimate_memory_requirements,
    find_crossover_point,
    run_scalability_experiment,
)
from src.sequential import process_images_sequential


@pytest.fixture
def test_images():
    """Generate two deterministic synthetic test images for numerical verification."""
    np.random.seed(42)
    img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    return [("sample_1.png", img1), ("sample_2.png", img2)]


@pytest.fixture
def filter_params():
    return {
        "kernel_size": 5,
        "sigma": 1.0,
        "repeats": 1,
    }


def test_optimized_parallel_matches_sequential_bit_exact(test_images, filter_params):
    """
    Verify zero-copy shared memory parallel implementation produces results that are
    100% bit-exact identical to the sequential baseline across all filter stages.
    """
    seq_results, seq_time = process_images_sequential(test_images, filter_params)
    opt_results, opt_time = process_images_parallel_optimized(
        test_images, filter_params, num_workers=2
    )

    assert len(seq_results) == len(opt_results)

    seq_map = {r["name"]: r for r in seq_results}
    opt_map = {r["name"]: r for r in opt_results}

    for name in seq_map:
        assert name in opt_map
        s_res = seq_map[name]
        o_res = opt_map[name]

        # Grayscale verification
        assert np.array_equal(s_res["gray"], o_res["gray"]), f"Grayscale mismatch in {name}"
        # Gaussian blur verification
        assert np.array_equal(s_res["blur"], o_res["blur"]), f"Blur mismatch in {name}"
        # Sobel edge detection verification
        assert np.array_equal(s_res["edges"], o_res["edges"]), f"Edges mismatch in {name}"


def test_shared_memory_batch_scope_cleanup(test_images, filter_params):
    """Verify shared_memory_batch_scope context manager allocates and unlinks segments cleanly without leaks."""
    saved_names = []
    with shared_memory_batch_scope(test_images, filter_params) as (task_list, shm_dict):
        assert len(task_list) == len(test_images)
        assert len(shm_dict) == 4
        saved_names = [shm.name for shm in shm_dict.values()]
        # While inside scope, segments exist
        for name in saved_names:
            shm_check = SharedMemory(name=name)
            shm_check.close()

    # Once outside scope, segments must be closed and unlinked
    for name in saved_names:
        with pytest.raises(FileNotFoundError):
            shm_check = SharedMemory(name=name)
            shm_check.close()


def test_persistent_worker_pool_lifecycle(test_images, filter_params):
    """Test PersistentWorkerPool warm startup, repeated execution, and clean shutdown."""
    pool = PersistentWorkerPool(num_workers=2)
    assert not pool.is_alive

    pool.start()
    assert pool.is_alive

    # First execution
    res1, t1 = process_images_parallel_optimized(
        test_images, filter_params, num_workers=2, persistent_pool=pool
    )
    assert len(res1) == 2
    assert t1 > 0

    # Second execution (reusing warm processes)
    res2, t2 = process_images_parallel_optimized(
        test_images, filter_params, num_workers=2, persistent_pool=pool
    )
    assert len(res2) == 2
    assert t2 > 0

    pool.shutdown()
    assert not pool.is_alive


def test_run_three_way_benchmark(test_images, filter_params):
    """Test 3-way comparative benchmark (Sequential vs Original vs Optimized)."""
    df, last_results, meta = run_three_way_benchmark(
        images=test_images,
        params=filter_params,
        worker_counts=[2],
        repeats=1,
        mode="steady_state",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Sequential, Original Parallel (2w), Optimized Parallel (2w)
    assert "mode" in df.columns
    assert "speedup" in df.columns
    assert "efficiency" in df.columns
    assert "optimization_gain" in df.columns
    assert "throughput_images_s" in df.columns

    modes = df["mode"].tolist()
    assert "Sequential" in modes
    assert "Original Parallel" in modes
    assert "Optimized Parallel" in modes

    # Sequential optimization gain is 1.0 by definition
    seq_row = df[df["mode"] == "Sequential"].iloc[0]
    assert seq_row["speedup"] == 1.0
    assert seq_row["optimization_gain"] == 1.0

    assert len(last_results) == 2
    assert "benchmark_mode" in meta


def test_memory_requirements_estimator():
    """Test peak RAM estimation and headroom calculations."""
    est = estimate_memory_requirements(count=16, width=1024, height=1024, num_workers=4)
    assert "estimated_peak_mb" in est
    assert "available_mb" in est
    assert "is_safe" in est
    assert est["estimated_peak_mb"] > 0
    assert isinstance(est["is_safe"], bool)


def test_find_crossover_point():
    """Test crossover point detection algorithm."""
    data = [
        {
            "workload_label": "2 imgs (512x512)",
            "num_images": 2,
            "total_megapixels": 0.5,
            "seq_time_s": 0.05,
            "orig_par_time_s": 0.09,
            "opt_par_time_s": 0.08,
            "opt_speedup": 0.625,
            "opt_gain": 1.12,
        },
        {
            "workload_label": "4 imgs (512x512)",
            "num_images": 4,
            "total_megapixels": 1.0,
            "seq_time_s": 0.10,
            "orig_par_time_s": 0.13,
            "opt_par_time_s": 0.11,
            "opt_speedup": 0.909,
            "opt_gain": 1.18,
        },
        {
            "workload_label": "8 imgs (512x512)",
            "num_images": 8,
            "total_megapixels": 2.0,
            "seq_time_s": 0.20,
            "orig_par_time_s": 0.22,
            "opt_par_time_s": 0.14,
            "opt_speedup": 1.428,
            "opt_gain": 1.57,
        },
    ]
    df = pd.DataFrame(data)
    crossover = find_crossover_point(df)
    assert crossover["crossover_observed"] is True
    assert crossover["crossover_images"] == 8
    assert crossover["speedup_at_crossover"] > 1.0


def test_scalability_experiment_run(filter_params):
    """Test small workload scalability sweep."""
    workload_configs = [
        {"label": "2 imgs (48x48)", "count": 2, "width": 48, "height": 48},
        {"label": "4 imgs (48x48)", "count": 4, "width": 48, "height": 48},
    ]
    sc_df, crossover_info = run_scalability_experiment(
        workload_configs=workload_configs,
        params=filter_params,
        num_workers=2,
        benchmark_mode="steady_state",
    )

    assert isinstance(sc_df, pd.DataFrame)
    assert len(sc_df) == 2
    assert "num_images" in sc_df.columns
    assert "seq_time_s" in sc_df.columns
    assert "opt_par_time_s" in sc_df.columns
    assert "opt_speedup" in sc_df.columns


def test_plots_optimized_generation(test_images, filter_params):
    """Test that all Module 11 specialized plotting routines return valid matplotlib Figures."""
    df, _, _ = run_three_way_benchmark(
        images=test_images,
        params=filter_params,
        worker_counts=[2],
        repeats=1,
        mode="steady_state",
    )

    fig1 = plot_three_way_execution_time(df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = plot_speedup_comparison(df)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    fig3 = plot_efficiency_comparison(df)
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    fig4 = plot_throughput_comparison(df)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    # Test crossover plot
    sc_df = pd.DataFrame([
        {
            "workload_label": "4 imgs",
            "num_images": 4,
            "total_megapixels": 1.0,
            "seq_time_s": 0.2,
            "orig_par_time_s": 0.22,
            "opt_par_time_s": 0.15,
            "opt_speedup": 1.33,
            "opt_gain": 1.46,
        },
        {
            "workload_label": "8 imgs",
            "num_images": 8,
            "total_megapixels": 2.0,
            "seq_time_s": 0.4,
            "orig_par_time_s": 0.38,
            "opt_par_time_s": 0.25,
            "opt_speedup": 1.60,
            "opt_gain": 1.52,
        },
    ])
    fig5 = plot_scalability_crossover(sc_df)
    assert isinstance(fig5, plt.Figure)
    plt.close(fig5)


def test_pdf_report_with_three_way_and_scalability(test_images, filter_params):
    """Verify that PDF generation seamlessly includes 3-way benchmark and scalability data."""
    df, last_results, _ = run_three_way_benchmark(
        images=test_images,
        params=filter_params,
        worker_counts=[2],
        repeats=1,
        mode="steady_state",
    )
    sc_df = pd.DataFrame([
        {
            "workload_label": "4 imgs",
            "num_images": 4,
            "total_megapixels": 1.0,
            "seq_time_s": 0.2,
            "orig_par_time_s": 0.22,
            "opt_par_time_s": 0.15,
            "opt_speedup": 1.33,
            "opt_gain": 1.46,
        },
        {
            "workload_label": "8 imgs",
            "num_images": 8,
            "total_megapixels": 2.0,
            "seq_time_s": 0.4,
            "orig_par_time_s": 0.38,
            "opt_par_time_s": 0.25,
            "opt_speedup": 1.60,
            "opt_gain": 1.52,
        },
    ])

    pdf_bytes = generate_benchmark_pdf(
        df=df,
        last_results=last_results,
        original_images=test_images,
        params=filter_params,
        scalability_df=sc_df,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
