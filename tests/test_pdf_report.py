"""
Tests for automated PDF benchmark report generation.
"""

import io
import pytest
import numpy as np
import pandas as pd

from src.pdf_report import generate_benchmark_pdf, _downsample_image_to_bytes, _generate_performance_analysis_text
from src.pipeline import process_single_image


def test_generate_benchmark_pdf_output_format(tiny_benchmark_df, tiny_images, params):
    # Checks that PDF generator returns valid in-memory bytes with %PDF- header
    last_results = [process_single_image((name, arr, params)) for name, arr in tiny_images]
    system_info = {
        "timestamp": "2026-11-25T20:00:00+05:30",
        "platform": "macOS-15-arm64",
        "machine": "arm64",
        "python_version": "3.12.16",
        "cpu_count": 10,
        "start_method": "spawn",
    }

    pdf_bytes = generate_benchmark_pdf(
        df=tiny_benchmark_df,
        last_results=last_results,
        original_images=tiny_images,
        params=params,
        system_info=system_info,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF-")


def test_downsample_image_aspect_ratio_and_resizing():
    # Checks that image downsampling strictly preserves aspect ratio and outputs JPEG bytes
    wide_arr = np.zeros((100, 500, 3), dtype=np.uint8)
    buf, orig_w, orig_h = _downsample_image_to_bytes(wide_arr, max_dim=250)
    assert orig_w == 500
    assert orig_h == 100
    assert isinstance(buf, io.BytesIO)

    tall_arr = np.zeros((400, 100), dtype=np.uint8)
    buf_tall, orig_wt, orig_ht = _downsample_image_to_bytes(tall_arr, max_dim=200)
    assert orig_wt == 100
    assert orig_ht == 400


def test_generate_performance_analysis_text(tiny_benchmark_df):
    # Checks that dynamic performance analysis generates key findings and paragraphs
    summary, analysis, metrics = _generate_performance_analysis_text(tiny_benchmark_df)
    assert len(summary) > 0
    assert len(analysis) > 0
    assert "best_workers" in metrics
    assert "speedup" in metrics


def test_generate_benchmark_pdf_sequential_only_case(tiny_benchmark_df, tiny_images, params):
    # Checks that PDF generator works gracefully even when only sequential results exist
    seq_only_df = tiny_benchmark_df[tiny_benchmark_df["mode"] == "Sequential"].copy()
    last_results = [process_single_image((name, arr, params)) for name, arr in tiny_images]

    pdf_bytes = generate_benchmark_pdf(
        df=seq_only_df,
        last_results=last_results,
        original_images=tiny_images,
        params=params,
        system_info={"cpu_count": 4},
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
