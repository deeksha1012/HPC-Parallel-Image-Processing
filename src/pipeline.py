"""
Single-image processing pipeline.

Defines the core processing workflow applied to each individual image:
1. Grayscale conversion
2. Gaussian blur
3. Sobel edge detection (on the blurred image)

This module provides a standalone, top-level function taking a single tuple argument
so that it can be cleanly pickled and dispatched to worker processes in parallel modules.
"""

import os
import time
from typing import Any
import numpy as np

from src.filters import gaussian_blur, sobel_edges, to_grayscale


def process_single_image(
    task: tuple[str, np.ndarray, dict[str, Any]]
) -> dict[str, Any]:
    """
    Process a single RGB image through grayscale, blur, and edge detection.

    Measures the execution time of the three filtering steps combined.
    Designed as a plain, top-level function taking a single tuple so it can be
    executed sequentially or sent across process boundaries via multiprocessing.

    Args:
        task (tuple): (name, rgb_array, params)
            - name (str): Identifier or filename of the image.
            - rgb_array (np.ndarray): 3D uint8 array with shape (H, W, 3).
            - params (dict): Filter settings containing:
                - 'kernel_size' (int): Blur kernel dimension.
                - 'sigma' (float): Blur standard deviation.

    Returns:
        dict[str, Any]: Dictionary containing:
            - 'name': Image name/identifier.
            - 'gray': 2D uint8 grayscale array.
            - 'blur': 2D uint8 Gaussian-blurred array.
            - 'edges': 2D uint8 Sobel edge map array.
            - 'time_seconds': Execution time in seconds for the 3 steps.
            - 'pid': Process ID that executed this task.
    """
    name, rgb_array, params = task
    kernel_size = params["kernel_size"]
    sigma = params["sigma"]

    # Start high-resolution wall-clock timer
    start_time = time.perf_counter()

    # Step 1: Convert color image to grayscale
    gray = to_grayscale(rgb_array)

    # Step 2: Apply Gaussian blur to reduce noise
    blur = gaussian_blur(gray, kernel_size=kernel_size, sigma=sigma)

    # Step 3: Compute Sobel edge gradients on the smoothed/blurred image
    edges = sobel_edges(blur)

    # Measure total processing time
    elapsed_time = time.perf_counter() - start_time

    return {
        "name": name,
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "time_seconds": elapsed_time,
        "pid": os.getpid(),
    }
