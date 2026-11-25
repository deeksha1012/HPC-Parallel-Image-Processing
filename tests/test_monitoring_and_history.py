"""
Tests for Module 10: HPC System Monitoring, Performance Intelligence, and SQLite History.
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.history import (
    clear_all_experiments,
    delete_experiment,
    export_history_to_csv_bytes,
    get_all_experiments,
    get_experiment_details,
    init_db,
    save_experiment,
)
from src.intelligence import analyze_performance, compute_throughput_metrics
from src.monitor import (
    ResourceMonitor,
    aggregate_worker_distribution,
    get_detailed_hardware_info,
    get_processor_name,
    plot_cpu_utilization,
    plot_memory_utilization,
    plot_worker_task_distribution,
)
from src.pdf_report import generate_benchmark_pdf


def test_hardware_detection_and_processor_name():
    """Verify hardware auto-detection returns valid cross-platform specifications."""
    info = get_detailed_hardware_info()
    proc_name = get_processor_name()

    assert isinstance(proc_name, str)
    assert len(proc_name) > 0

    assert "processor" in info
    assert "machine" in info
    assert "platform" in info
    assert "physical_cores" in info
    assert "logical_cores" in info
    assert "total_ram_gb" in info
    assert "available_ram_gb" in info
    assert "ram_percent" in info
    assert "python_version" in info

    assert info["physical_cores"] >= 1
    assert info["logical_cores"] >= info["physical_cores"] or info["logical_cores"] >= 1
    assert info["total_ram_gb"] > 0
    assert info["available_ram_gb"] >= 0
    assert 0.0 <= info["ram_percent"] <= 100.0


def test_resource_monitor_thread_sampling_and_summary():
    """Verify ResourceMonitor background sampling, DataFrame conversion, and summary statistics."""
    monitor = ResourceMonitor(interval=0.05)
    assert not monitor.is_running()

    monitor.start()
    assert monitor.is_running()

    # Allow several samples to be collected
    time.sleep(0.2)

    monitor.stop()
    assert not monitor.is_running()

    history_df = monitor.get_history_df()
    assert isinstance(history_df, pd.DataFrame)
    assert not history_df.empty
    assert "time_s" in history_df.columns
    assert "cpu_percent" in history_df.columns
    assert "memory_percent" in history_df.columns
    assert "memory_used_mb" in history_df.columns

    summary = monitor.get_summary()
    assert isinstance(summary, dict)
    assert summary["samples_count"] >= 1
    assert summary["duration_seconds"] > 0
    assert summary["avg_cpu_percent"] >= 0.0
    assert summary["peak_cpu_percent"] >= 0.0
    assert summary["peak_memory_mb"] > 0.0
    assert "high_memory_warning" in summary


def test_worker_task_distribution_and_plots():
    """Verify worker PID aggregation and Matplotlib visualization functions."""
    # Synthetic results representing 4 tasks handled by 2 different worker PIDs
    dummy_results = [
        {"name": "img1.png", "time_seconds": 0.12, "pid": 1001},
        {"name": "img2.png", "time_seconds": 0.14, "pid": 1002},
        {"name": "img3.png", "time_seconds": 0.11, "pid": 1001},
        {"name": "img4.png", "time_seconds": 0.13, "pid": 1002},
    ]

    agg_df = aggregate_worker_distribution(dummy_results)
    assert len(agg_df) == 2
    assert set(agg_df["pid"]) == {1001, 1002}
    assert agg_df["tasks_processed"].sum() == 4
    assert np.isclose(agg_df["task_share_pct"].sum(), 100.0)

    # Test plotting functions return valid figures
    fig_w = plot_worker_task_distribution(dummy_results)
    assert fig_w is not None

    # Test telemetry plotting
    dummy_history = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "cpu_percent": [15.0, 45.0, 70.0],
            "core_0": [20.0, 50.0, 80.0],
            "core_1": [10.0, 40.0, 60.0],
            "memory_used_mb": [500.0, 520.0, 530.0],
            "memory_percent": [30.0, 31.0, 31.5],
        }
    )
    fig_cpu = plot_cpu_utilization(dummy_history)
    fig_mem = plot_memory_utilization(dummy_history)
    assert fig_cpu is not None
    assert fig_mem is not None


def test_performance_intelligence_rule_based_engine(tiny_benchmark_df):
    """Verify throughput calculation and performance intelligence analytical recommendations."""
    # 1. Throughput metrics augmentation
    aug_df = compute_throughput_metrics(tiny_benchmark_df)
    assert "throughput_images_per_s" in aug_df.columns
    assert "throughput_mp_per_s" in aug_df.columns
    assert "cv_percent" in aug_df.columns
    assert "ideal_speedup" in aug_df.columns
    assert "speedup_deficit" in aug_df.columns
    assert (aug_df["throughput_images_per_s"] > 0).all()

    # 2. Performance intelligence evaluation
    dummy_summary = {
        "duration_seconds": 1.5,
        "samples_count": 8,
        "avg_cpu_percent": 65.0,
        "peak_cpu_percent": 88.0,
        "baseline_memory_mb": 400.0,
        "peak_memory_mb": 450.0,
        "memory_growth_mb": 50.0,
        "peak_memory_percent": 35.0,
        "high_memory_warning": False,
    }
    dummy_hw = {
        "processor": "Apple M4",
        "machine": "arm64",
        "platform": "macOS",
        "physical_cores": 10,
        "logical_cores": 10,
    }

    intel = analyze_performance(tiny_benchmark_df, dummy_summary, dummy_hw)
    assert "fastest_overall" in intel
    assert "fastest_parallel" in intel
    assert "recommended_workers" in intel
    assert "recommendation_reason" in intel
    assert "key_takeaways" in intel
    assert len(intel["key_takeaways"]) > 0


def test_sqlite_benchmark_history_operations(tmp_path, tiny_benchmark_df):
    """Verify SQLite history initialization, record insertion, querying, export, and deletion."""
    db_file = tmp_path / "test_history.db"

    # 1. Initialize DB
    init_db(db_file)
    assert db_file.exists()

    # 2. Save experiment
    params = {"kernel_size": 15, "sigma": 3.0, "repeats": 2}
    exp_id = save_experiment(
        df=tiny_benchmark_df,
        params=params,
        num_images=2,
        total_megapixels=0.5,
        monitor_summary={"avg_cpu_percent": 55.0, "peak_memory_mb": 300.0},
        notes="Pytest unit trial",
        db_path=db_file,
    )
    assert exp_id > 0

    # 3. Retrieve all experiments
    all_df = get_all_experiments(db_file)
    assert len(all_df) == 1
    assert all_df.iloc[0]["id"] == exp_id
    assert all_df.iloc[0]["num_images"] == 2
    assert all_df.iloc[0]["fastest_mode"] in ["Sequential", "Parallel"]

    # 4. Retrieve single experiment details
    details = get_experiment_details(exp_id, db_file)
    assert details is not None
    assert details["id"] == exp_id
    assert isinstance(details["metrics_df"], pd.DataFrame)
    assert len(details["metrics_df"]) == len(tiny_benchmark_df)
    assert details["monitor_summary"]["avg_cpu_percent"] == 55.0

    # 5. Export to CSV bytes
    csv_bytes = export_history_to_csv_bytes(db_file)
    assert isinstance(csv_bytes, bytes)
    assert b"fastest_mode" in csv_bytes

    # 6. Delete single experiment
    deleted = delete_experiment(exp_id, db_file)
    assert deleted is True
    assert len(get_all_experiments(db_file)) == 0

    # 7. Clear all experiments
    save_experiment(tiny_benchmark_df, params, 2, 0.5, db_path=db_file)
    assert len(get_all_experiments(db_file)) == 1
    cleared = clear_all_experiments(db_file)
    assert cleared is True
    assert len(get_all_experiments(db_file)) == 0


def test_pdf_report_with_module_10_telemetry(tiny_images, tiny_benchmark_df, params):
    """Verify that generate_benchmark_pdf accepts Module 10 telemetry and produces valid PDF bytes."""
    dummy_results = [
        {
            "name": name,
            "gray": np.zeros((30, 30), dtype=np.uint8),
            "blur": np.zeros((30, 30), dtype=np.uint8),
            "edges": np.zeros((30, 30), dtype=np.uint8),
            "time_seconds": 0.05,
            "pid": 5001,
        }
        for name, _ in tiny_images
    ]

    dummy_summary = {
        "duration_seconds": 0.85,
        "samples_count": 5,
        "avg_cpu_percent": 42.5,
        "peak_cpu_percent": 75.0,
        "baseline_memory_mb": 350.0,
        "peak_memory_mb": 390.0,
        "memory_growth_mb": 40.0,
        "peak_memory_percent": 25.0,
        "high_memory_warning": False,
    }

    dummy_history = pd.DataFrame(
        {
            "time_s": [0.0, 0.2, 0.4, 0.6],
            "cpu_percent": [10.0, 35.0, 65.0, 20.0],
            "core_0": [15.0, 40.0, 70.0, 25.0],
            "core_1": [5.0, 30.0, 60.0, 15.0],
            "memory_used_mb": [350.0, 365.0, 380.0, 375.0],
            "memory_percent": [22.0, 23.0, 24.0, 23.5],
        }
    )

    dummy_sys_info = get_detailed_hardware_info()
    intel = analyze_performance(tiny_benchmark_df, dummy_summary, dummy_sys_info)

    pdf_bytes = generate_benchmark_pdf(
        df=tiny_benchmark_df,
        last_results=dummy_results,
        original_images=tiny_images,
        params=params,
        system_info=dummy_sys_info,
        monitor_summary=dummy_summary,
        monitor_history=dummy_history,
        intelligence=intel,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_streamlit_app_with_active_benchmark_data(tiny_images, tiny_benchmark_df, params):
    """Verify that Streamlit app renders all 6 tabs without exceptions when benchmark and telemetry data are present."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()

    dummy_results = [
        {
            "name": name,
            "gray": np.zeros((30, 30), dtype=np.uint8),
            "blur": np.zeros((30, 30), dtype=np.uint8),
            "edges": np.zeros((30, 30), dtype=np.uint8),
            "time_seconds": 0.05,
            "pid": 5001,
        }
        for name, _ in tiny_images
    ]
    at.session_state["loaded_images"] = tiny_images
    at.session_state["benchmark_df"] = tiny_benchmark_df
    at.session_state["last_results"] = dummy_results
    at.session_state["original_images"] = tiny_images
    at.session_state["params"] = params
    at.session_state["system_info"] = get_detailed_hardware_info()
    at.session_state["monitor_summary"] = {
        "duration_seconds": 0.5,
        "samples_count": 3,
        "avg_cpu_percent": 30.0,
        "peak_cpu_percent": 60.0,
        "baseline_memory_mb": 200.0,
        "peak_memory_mb": 220.0,
        "memory_growth_mb": 20.0,
        "peak_memory_percent": 25.0,
        "high_memory_warning": False,
    }
    at.session_state["monitor_history"] = pd.DataFrame(
        {
            "time_s": [0.0, 0.2, 0.4],
            "cpu_percent": [20.0, 40.0, 50.0],
            "core_0": [25.0, 45.0, 55.0],
            "memory_used_mb": [200.0, 210.0, 220.0],
            "memory_percent": [24.0, 24.5, 25.0],
        }
    )
    at.session_state["intelligence"] = analyze_performance(tiny_benchmark_df)
    at.run()
    assert not at.exception
