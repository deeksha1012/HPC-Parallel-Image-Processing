"""
Pure NumPy implementations of core image processing filters.

All filters operate directly on NumPy arrays using vectorized array slicing
and arithmetic for high performance without per-pixel Python loops.
Implementations:
1. to_grayscale: ITU-R BT.601 luminance conversion
2. gaussian_kernel_1d: 1D Gaussian probability distribution function
3. gaussian_blur: 2-pass separable 1D convolution (horizontal then vertical)
4. sobel_edges: 3x3 Sobel gradient operator with linear normalization
"""

import numpy as np


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB color image to grayscale using standard luminance weights.

    Formula: Y = 0.299 * R + 0.587 * G + 0.114 * B (ITU-R BT.601 recommendation)
    Human vision is most sensitive to green, moderately to red, and least to blue.

    Args:
        rgb (np.ndarray): 3D uint8 array of shape (Height, Width, 3).

    Returns:
        np.ndarray: 2D uint8 array of shape (Height, Width).
    """
    # Extract color channels as float32 to prevent overflow during arithmetic
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # Weighted sum according to human perceptual sensitivity
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Round to nearest integer, clamp to [0, 255], and cast back to uint8
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Generate a 1-dimensional normalized Gaussian filter kernel.

    Formula: G(x) = exp(-x^2 / (2 * sigma^2)), normalized so sum(G) == 1.0.

    Args:
        kernel_size (int): Size of the kernel window. Must be odd and >= 3.
        sigma (float): Standard deviation controlling blur spread. Must be > 0.

    Returns:
        np.ndarray: 1D float32 array of shape (kernel_size,) summing to 1.0.

    Raises:
        ValueError: If kernel_size is even, < 3, or if sigma <= 0.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be an odd integer >= 3, got {kernel_size}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be a positive number (> 0), got {sigma}")

    radius = kernel_size // 2
    # Sample points centered around 0: [-radius, ..., 0, ..., radius]
    x = np.arange(-radius, radius + 1, dtype=np.float32)

    # Compute unnormalized Gaussian values
    kernel = np.exp(-0.5 * (x / sigma) ** 2)

    # Normalize so all weights sum exactly to 1.0 (preserves image brightness)
    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)


def gaussian_blur(gray: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using separable 1D convolution (horizontal then vertical).

    Instead of a 2D convolution requiring K^2 operations per pixel, a separable
    filter performs two 1D passes requiring only 2*K operations per pixel.
    Boundary pixels are handled using edge-padding (replicates border values).

    Args:
        gray (np.ndarray): 2D uint8 grayscale image array of shape (Height, Width).
        kernel_size (int): Blur window size (odd integer >= 3).
        sigma (float): Gaussian standard deviation (> 0).

    Returns:
        np.ndarray: 2D uint8 blurred image array of shape (Height, Width).
    """
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    radius = kernel_size // 2
    h, w = gray.shape

    # Convert input to float32 for high precision convolution
    gray_float = gray.astype(np.float32)

    # -------------------------------------------------------------------------
    # Pass 1: Horizontal 1D convolution across columns
    # -------------------------------------------------------------------------
    # Pad left and right borders by 'radius' pixels using edge replication
    padded_h = np.pad(gray_float, ((0, 0), (radius, radius)), mode="edge")
    blurred_h = np.zeros((h, w), dtype=np.float32)

    # Accumulate kernel tap contributions using vectorized array slicing
    for i in range(kernel_size):
        blurred_h += kernel[i] * padded_h[:, i : i + w]

    # -------------------------------------------------------------------------
    # Pass 2: Vertical 1D convolution across rows
    # -------------------------------------------------------------------------
    # Pad top and bottom borders by 'radius' pixels using edge replication
    padded_v = np.pad(blurred_h, ((radius, radius), (0, 0)), mode="edge")
    blurred_v = np.zeros((h, w), dtype=np.float32)

    for j in range(kernel_size):
        blurred_v += kernel[j] * padded_v[j : j + h, :]

    # Round to nearest integer, clamp to [0, 255], and cast back to uint8
    return np.clip(np.rint(blurred_v), 0, 255).astype(np.uint8)


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """
    Compute image edges using the 3x3 Sobel gradient operator.

    Calculates horizontal gradients (Gx) and vertical gradients (Gy) using
    standard Sobel convolution kernels implemented via vectorized array slices.
    The gradient magnitude is computed as sqrt(Gx^2 + Gy^2) and linearly scaled
    so that the maximum edge intensity is mapped to 255.

    Sobel Kernels:
         Gx = [[-1, 0, +1],       Gy = [[-1, -2, -1],
               [-2, 0, +2],             [ 0,  0,  0],
               [-1, 0, +1]]             [+1, +2, +1]]

    Args:
        gray (np.ndarray): 2D uint8 grayscale image array of shape (Height, Width).

    Returns:
        np.ndarray: 2D uint8 edge intensity map of shape (Height, Width).
    """
    h, w = gray.shape

    # Pad by 1 pixel on all sides to allow 3x3 kernel evaluation at image borders
    padded = np.pad(gray.astype(np.float32), ((1, 1), (1, 1)), mode="edge")

    # Extract the 8 neighboring pixel slices of shape (h, w)
    top_left     = padded[0:h, 0:w]
    top_center   = padded[0:h, 1:w + 1]
    top_right    = padded[0:h, 2:w + 2]
    mid_left     = padded[1:h + 1, 0:w]
    mid_right    = padded[1:h + 1, 2:w + 2]
    bottom_left  = padded[2:h + 2, 0:w]
    bottom_center= padded[2:h + 2, 1:w + 1]
    bottom_right = padded[2:h + 2, 2:w + 2]

    # Compute horizontal gradient (Gx) - detects vertical boundaries
    gx = (top_right - top_left) + 2.0 * (mid_right - mid_left) + (bottom_right - bottom_left)

    # Compute vertical gradient (Gy) - detects horizontal boundaries
    gy = (bottom_left - top_left) + 2.0 * (bottom_center - top_center) + (bottom_right - top_right)

    # Euclidean gradient magnitude
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Linearly scale edge magnitude so the maximum intensity is 255
    max_val = np.max(magnitude)
    if max_val > 0.0:
        scaled = (magnitude / max_val) * 255.0
    else:
        scaled = np.zeros_like(magnitude)

    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
