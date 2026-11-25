"""
Automated unit tests for Module 9: UI components, helper utilities, and in-memory sample generator.
"""

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.sample_generator import (
    RESOLUTION_PRESETS,
    generate_sample_images_in_memory,
)
from src.ui.components import (
    array_to_base64_jpeg,
    format_bytes_size,
)


def test_generate_sample_images_in_memory_default():
    # Checks that in-memory sample generator produces correct count, shape, and uint8 type
    count = 3
    w, h = 400, 300
    images = generate_sample_images_in_memory(count=count, width=w, height=h, seed=10)

    assert len(images) == count
    for name, arr in images:
        assert isinstance(name, str)
        assert arr.shape == (h, w, 3)
        assert arr.dtype == np.uint8


def test_generate_sample_images_in_memory_bounds_and_presets():
    # Checks that count and dimension arguments are clamped to safe memory limits
    # Over-limit count clamped to 16
    oversized = generate_sample_images_in_memory(count=50, width=3000, height=3000)
    assert len(oversized) == 16
    assert oversized[0][1].shape[0] == 1800  # Clamped height
    assert oversized[0][1].shape[1] == 2400  # Clamped width

    # Under-limit count clamped to 1
    undersized = generate_sample_images_in_memory(count=0, width=50, height=50)
    assert len(undersized) == 1
    assert undersized[0][1].shape[0] == 150
    assert undersized[0][1].shape[1] == 200

    # Validate that standard resolution presets exist and have valid aspect ratios
    assert "Standard (1200 × 900 px)" in RESOLUTION_PRESETS
    for label, (pw, ph) in RESOLUTION_PRESETS.items():
        assert pw > 0 and ph > 0


def test_generate_sample_images_reproducibility():
    # Checks that identical seeds yield identical image pixel arrays
    run1 = generate_sample_images_in_memory(count=2, width=300, height=200, seed=77)
    run2 = generate_sample_images_in_memory(count=2, width=300, height=200, seed=77)

    assert np.array_equal(run1[0][1], run2[0][1])
    assert np.array_equal(run1[1][1], run2[1][1])


def test_format_bytes_size_utilities():
    # Checks conversion of raw byte counts to human-readable strings
    assert format_bytes_size(500) == "500 B"
    assert format_bytes_size(1024) == "1.0 KB"
    assert format_bytes_size(2048 * 1024) == "2.00 MB"


def test_array_to_base64_jpeg():
    # Checks that 2D grayscale and 3D RGB arrays convert to non-empty base64 strings
    gray_arr = np.zeros((50, 60), dtype=np.uint8)
    b64_gray = array_to_base64_jpeg(gray_arr)
    assert isinstance(b64_gray, str)
    assert len(b64_gray) > 100

    rgb_arr = np.ones((50, 60, 3), dtype=np.uint8) * 128
    b64_rgb = array_to_base64_jpeg(rgb_arr)
    assert isinstance(b64_rgb, str)
    assert len(b64_rgb) > 100


def test_streamlit_app_renders_without_exceptions():
    # Verifies that app.py compiles, initializes tabs, and renders with 0 exceptions
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()
    assert not at.exception
