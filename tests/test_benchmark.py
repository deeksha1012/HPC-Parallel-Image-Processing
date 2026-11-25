"""
Tests for benchmark execution, DataFrame structure, and input validation.
"""

import pytest
import numpy as np
from src.benchmark import run_benchmark


def test_benchmark_dataframe_structure_and_metrics(tiny_benchmark_df):
    # Checks exact column names, row count, positive times, and speedup/efficiency mathematical consistency
    expected_cols = [
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
    assert list(tiny_benchmark_df.columns) == expected_cols
    assert len(tiny_benchmark_df) == 2

    seq_row = tiny_benchmark_df[tiny_benchmark_df["mode"] == "Sequential"].iloc[0]
    par_row = tiny_benchmark_df[tiny_benchmark_df["mode"] == "Parallel"].iloc[0]

    assert seq_row["mean_time_s"] > 0
    assert par_row["mean_time_s"] > 0

    expected_speedup = seq_row["mean_time_s"] / par_row["mean_time_s"]
    assert np.isclose(par_row["speedup"], expected_speedup, atol=1e-5)

    expected_efficiency = par_row["speedup"] / par_row["workers"]
    assert np.isclose(par_row["efficiency"], expected_efficiency, atol=1e-5)


def test_benchmark_validation_errors(tiny_images, params):
    # Checks that run_benchmark raises ValueError for zero workers or zero repeats
    with pytest.raises(ValueError, match="Worker count"):
        run_benchmark(tiny_images, params, worker_counts=[0], repeats=1)

    with pytest.raises(ValueError, match="repeats"):
        run_benchmark(tiny_images, params, worker_counts=[2], repeats=0)
