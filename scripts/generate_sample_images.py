"""
Generate synthetic sample images for benchmarking image processing algorithms.

This script uses only NumPy and Pillow to generate reproducible test images
with smooth gradients, geometric shapes (rectangles and circles), and subtle noise.
No internet connection or external image downloads are required.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def generate_single_image(width: int, height: int, seed: int) -> Image.Image:
    """
    Generate a single synthetic RGB image using a specific random seed.

    Steps:
    1. Build a smooth 2-color diagonal gradient background using NumPy arrays.
    2. Draw random filled rectangles and circles using Pillow's ImageDraw.
    3. Add a small amount of random Gaussian noise so filtering produces visible changes.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        seed (int): Random seed for reproducible generation.

    Returns:
        PIL.Image.Image: The resulting synthetic RGB image.
    """
    # Create an independent random number generator for this image
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------
    # Step 1: Create a smooth diagonal color gradient background
    # -------------------------------------------------------------
    # Pick two random RGB base colors (range 30 to 220 for pleasant contrast)
    color_start = rng.integers(30, 220, size=3, dtype=np.int32)
    color_end = rng.integers(30, 220, size=3, dtype=np.int32)

    # Create coordinate grids from 0.0 to 1.0 along width (x) and height (y)
    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Average x and y grids to create a smooth diagonal transition factor [0.0, 1.0]
    blend_factor = (grid_x + grid_y) / 2.0
    # Add a third dimension so it can broadcast across the 3 RGB color channels
    blend_factor = blend_factor[:, :, np.newaxis]

    # Linearly interpolate between the two colors across the image canvas
    gradient_array = (1.0 - blend_factor) * color_start + blend_factor * color_end
    # Ensure values remain strictly within the 8-bit unsigned integer range [0, 255]
    gradient_uint8 = np.clip(gradient_array, 0, 255).astype(np.uint8)

    # Convert the NumPy gradient array into a Pillow Image to draw vector shapes
    image = Image.fromarray(gradient_uint8, mode="RGB")
    draw = ImageDraw.Draw(image)

    # -------------------------------------------------------------
    # Step 2: Draw random geometric shapes (rectangles and circles)
    # -------------------------------------------------------------
    # Draw between 6 and 10 random filled rectangles
    num_rectangles = int(rng.integers(6, 11))
    for _ in range(num_rectangles):
        # Choose upper-left corner
        x1 = int(rng.integers(0, max(1, width - 100)))
        y1 = int(rng.integers(0, max(1, height - 100)))
        # Choose rectangle dimensions
        rect_width = int(rng.integers(60, min(500, width - x1)))
        rect_height = int(rng.integers(60, min(500, height - y1)))
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        rect_color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        draw.rectangle([x1, y1, x2, y2], fill=rect_color)

    # Draw between 6 and 10 random filled circles
    num_circles = int(rng.integers(6, 11))
    max_radius = min(250, min(width, height) // 3)
    for _ in range(num_circles):
        radius = int(rng.integers(30, max_radius))
        center_x = int(rng.integers(radius, max(radius + 1, width - radius)))
        center_y = int(rng.integers(radius, max(radius + 1, height - radius)))
        circle_color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        bounding_box = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]
        draw.ellipse(bounding_box, fill=circle_color)

    # -------------------------------------------------------------
    # Step 3: Add slight random noise
    # -------------------------------------------------------------
    # Convert image back to an array for pixel manipulation
    image_array = np.array(image, dtype=np.float32)

    # Generate Gaussian noise (mean = 0, standard deviation = 10)
    noise = rng.normal(loc=0.0, scale=10.0, size=image_array.shape).astype(np.float32)

    # Add noise to original image and clamp pixel values to [0, 255]
    noisy_array = np.clip(image_array + noise, 0.0, 255.0).astype(np.uint8)

    # Return final Pillow Image
    return Image.fromarray(noisy_array, mode="RGB")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for sample image generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample test images for parallel image processing benchmarks."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of synthetic sample images to generate (default: 12)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2000,
        help="Width of each generated image in pixels (default: 2000)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1500,
        help="Height of each generated image in pixels (default: 1500)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="sample_images",
        help="Directory to save generated sample images (default: sample_images)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible image generation (default: 42)",
    )
    return parser.parse_args()


def main():
    """Main execution flow for generating and saving sample images."""
    args = parse_arguments()

    # Ensure output directory exists (create it if missing)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} sample images in '{output_dir}'...")

    for index in range(args.count):
        # Use seed + index so each image has a distinct layout, but runs are fully reproducible
        current_seed = args.seed + index
        img = generate_single_image(args.width, args.height, current_seed)

        # File naming pattern: sample_01.jpg, sample_02.jpg, ...
        filename = f"sample_{index + 1:02d}.jpg"
        filepath = output_dir / filename

        # Save as JPEG with quality 90
        img.save(filepath, format="JPEG", quality=90)
        print(f"Saved: {filename} ({args.width}x{args.height} pixels)")

    print("All sample images generated successfully.")


if __name__ == "__main__":
    main()
