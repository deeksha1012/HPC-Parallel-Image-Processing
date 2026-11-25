"""
Tests for image pipeline execution, multiprocessing consistency, and image I/O.
"""

import io
import pytest
import numpy as np
from PIL import Image

from src.image_io import load_image
from src.parallel import process_images_parallel
from src.pipeline import process_single_image
from src.sequential import process_images_sequential


def test_process_single_image_output_structure(params):
    # Checks process_single_image dictionary keys, array shapes, and non-negative execution time
    h, w = 32, 48
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    task = ("test_img.png", rgb, params)
    res = process_single_image(task)

    assert res["name"] == "test_img.png"
    assert res["gray"].shape == (h, w) and res["gray"].dtype == np.uint8
    assert res["blur"].shape == (h, w) and res["blur"].dtype == np.uint8
    assert res["edges"].shape == (h, w) and res["edges"].dtype == np.uint8
    assert res["time_seconds"] >= 0.0
    assert isinstance(res["pid"], int)


def test_sequential_and_parallel_consistency(tiny_images, params):
    # Checks that sequential and parallel execution produce identical images in identical order
    seq_results, _ = process_images_sequential(tiny_images, params)
    par_results, _ = process_images_parallel(tiny_images, params, num_workers=2)

    assert len(seq_results) == len(par_results)
    for s_res, p_res in zip(seq_results, par_results):
        assert s_res["name"] == p_res["name"]
        assert np.array_equal(s_res["gray"], p_res["gray"])
        assert np.array_equal(s_res["blur"], p_res["blur"])
        assert np.array_equal(s_res["edges"], p_res["edges"])


def test_parallel_validation_errors(tiny_images, params):
    # Checks that process_images_parallel raises ValueError for 0 workers or empty image lists
    with pytest.raises(ValueError, match="num_workers"):
        process_images_parallel(tiny_images, params, num_workers=0)

    with pytest.raises(ValueError, match="empty"):
        process_images_parallel([], params, num_workers=2)


def test_load_image_alpha_conversion_and_invalid_bytes():
    # Checks that RGBA in-memory images convert to (H, W, 3) RGB and invalid bytes raise an error
    rgba_img = Image.new("RGBA", (40, 30), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    rgba_img.save(buf, format="PNG")
    buf.seek(0)

    loaded = load_image(buf)
    assert loaded.shape == (30, 40, 3)
    assert loaded.dtype == np.uint8

    with pytest.raises(Exception):
        load_image(io.BytesIO(b"not an image file"))
