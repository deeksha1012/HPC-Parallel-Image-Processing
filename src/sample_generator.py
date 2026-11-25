"""
In-memory synthetic sample image generator for the Streamlit UI and test pipelines.

Reuses the reproducible synthetic image generator from scripts.generate_sample_images
to generate test datasets with smooth color gradients, vector shapes, and subtle noise
without writing temporary files to disk.
"""

from typing import Dict, List, Tuple
import numpy as np

from scripts.generate_sample_images import generate_single_image

# Pre-configured standard resolution presets for quick classroom and benchmark testing
RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "Fast Test (800 × 600 px)": (800, 600),
    "Standard (1200 × 900 px)": (1200, 900),
    "High Definition (1600 × 1200 px)": (1600, 1200),
    "Full Benchmark (2000 × 1500 px)": (2000, 1500),
}


def generate_sample_images_in_memory(
    count: int = 4,
    width: int = 1200,
    height: int = 900,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate synthetic sample images in memory as (filename, RGB_uint8_array) pairs.

    Args:
        count: Number of images to generate (constrained between 1 and 16 for memory safety).
        width: Image width in pixels (constrained between 200 and 2400).
        height: Image height in pixels (constrained between 150 and 1800).
        seed: Base random seed for reproducible image generation.

    Returns:
        List[Tuple[str, np.ndarray]]: List of (filename, 3D uint8 array) tuples.
    """
    # Enforce safe bounds for memory and responsiveness
    safe_count = max(1, min(int(count), 16))
    safe_width = max(200, min(int(width), 2400))
    safe_height = max(150, min(int(height), 1800))

    generated_images = []
    for i in range(safe_count):
        image_seed = seed + i
        pil_img = generate_single_image(width=safe_width, height=safe_height, seed=image_seed)
        rgb_arr = np.array(pil_img, dtype=np.uint8)
        filename = f"sample_{i + 1:02d}.jpg"
        generated_images.append((filename, rgb_arr))

    return generated_images
