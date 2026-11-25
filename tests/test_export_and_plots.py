"""
Tests for in-memory exports (PNG, CSV, ZIP) and Matplotlib chart generation.
"""

import io
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.export import (
    create_results_zip,
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
    image_to_png_bytes,
)
from src.plots import plot_efficiency, plot_execution_time, plot_speedup


def test_in_memory_export_utilities(tiny_benchmark_df):
    # Checks PNG binary signature, DataFrame CSV serialization, and ZIP archive contents
    test_arr = np.zeros((20, 20), dtype=np.uint8)
    png_bytes = image_to_png_bytes(test_arr)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    csv_bytes = dataframe_to_csv_bytes(tiny_benchmark_df)
    reloaded_df = pd.read_csv(io.BytesIO(csv_bytes))
    assert reloaded_df.shape == tiny_benchmark_df.shape

    sample_results = [
        {
            "name": "sample_01.jpg",
            "gray": np.zeros((10, 10), dtype=np.uint8),
            "blur": np.zeros((10, 10), dtype=np.uint8),
            "edges": np.zeros((10, 10), dtype=np.uint8),
        }
    ]
    zip_bytes = create_results_zip(sample_results, which=("gray", "blur", "edges"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert "gray/sample_01_gray.png" in namelist
        assert "blur/sample_01_blur.png" in namelist
        assert "edges/sample_01_edges.png" in namelist


def test_plot_generation_and_figure_export(tiny_benchmark_df):
    # Checks that plot functions return valid figures with axes and serialize to PNG bytes
    fig_time = plot_execution_time(tiny_benchmark_df)
    assert len(fig_time.axes) > 0
    bytes_time = figure_to_png_bytes(fig_time)
    assert len(bytes_time) > 0
    plt.close(fig_time)

    fig_speedup = plot_speedup(tiny_benchmark_df)
    assert len(fig_speedup.axes) > 0
    bytes_speedup = figure_to_png_bytes(fig_speedup)
    assert len(bytes_speedup) > 0
    plt.close(fig_speedup)

    fig_eff = plot_efficiency(tiny_benchmark_df)
    assert len(fig_eff.axes) > 0
    bytes_eff = figure_to_png_bytes(fig_eff)
    assert len(bytes_eff) > 0
    plt.close(fig_eff)
