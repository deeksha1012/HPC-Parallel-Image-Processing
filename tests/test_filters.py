"""
Tests for pure-NumPy image filters: grayscale, Gaussian blur, and Sobel edge detection.
"""

import pytest
import numpy as np

from src.filters import gaussian_blur, gaussian_kernel_1d, sobel_edges, to_grayscale


def test_grayscale_weights_and_shape():
    # Checks luminance weights on primary colors and confirms 2D non-square uint8 output
    h, w = 12, 18
    red = np.zeros((h, w, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    green = np.zeros((h, w, 3), dtype=np.uint8)
    green[:, :, 1] = 255
    blue = np.zeros((h, w, 3), dtype=np.uint8)
    blue[:, :, 2] = 255
    white = np.full((h, w, 3), 255, dtype=np.uint8)

    gray_red = to_grayscale(red)
    gray_green = to_grayscale(green)
    gray_blue = to_grayscale(blue)
    gray_white = to_grayscale(white)

    assert gray_red.shape == (h, w) and gray_red.dtype == np.uint8
    assert np.all(gray_red == 76)
    assert np.all(gray_green == 150)
    assert np.all(gray_blue == 29)
    assert np.all(gray_white == 255)


def test_gaussian_kernel_1d_properties_and_validation():
    # Checks kernel normalization, symmetry, length, and error handling for invalid arguments
    k_size = 7
    sigma = 1.5
    kernel = gaussian_kernel_1d(k_size, sigma)

    assert kernel.dtype == np.float32
    assert len(kernel) == k_size
    assert np.isclose(np.sum(kernel), 1.0, atol=1e-5)
    assert np.allclose(kernel, kernel[::-1])

    with pytest.raises(ValueError, match="odd integer"):
        gaussian_kernel_1d(6, 1.0)
    with pytest.raises(ValueError, match="odd integer"):
        gaussian_kernel_1d(2, 1.0)
    with pytest.raises(ValueError, match="positive number"):
        gaussian_kernel_1d(5, 0.0)
    with pytest.raises(ValueError, match="positive number"):
        gaussian_kernel_1d(5, -1.0)


def test_gaussian_blur_behavior_and_shape():
    # Checks that flat images are unchanged and noise standard deviation decreases after blur
    flat = np.full((30, 40), 128, dtype=np.uint8)
    blurred_flat = gaussian_blur(flat, kernel_size=5, sigma=1.0)
    assert blurred_flat.shape == flat.shape and blurred_flat.dtype == np.uint8
    assert np.array_equal(blurred_flat, flat)

    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(30, 40), dtype=np.uint8)
    blurred_noise = gaussian_blur(noise, kernel_size=5, sigma=1.5)
    assert blurred_noise.shape == noise.shape and blurred_noise.dtype == np.uint8
    assert np.std(blurred_noise) < np.std(noise)


def test_sobel_edges_boundary_detection_and_shape():
    # Checks zero edges on uniform images and correct edge activation across a vertical step
    flat = np.full((30, 40), 100, dtype=np.uint8)
    edges_flat = sobel_edges(flat)
    assert edges_flat.shape == flat.shape and edges_flat.dtype == np.uint8
    assert np.all(edges_flat == 0)

    step = np.zeros((30, 40), dtype=np.uint8)
    step[:, 20:] = 255
    edges_step = sobel_edges(step)
    assert edges_step.shape == step.shape and edges_step.dtype == np.uint8
    assert np.any(edges_step[:, 19:21] > 0)
    assert np.all(edges_step[:, :5] == 0)
    assert np.all(edges_step[:, 35:] == 0)
