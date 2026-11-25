"""
Sequential batch image processor.

Processes multiple images sequentially using a single CPU core. Serves as
the single-threaded baseline (T1) against which parallel speedup and efficiency
are measured.
"""

import argparse
import time
from pathlib import Path
from typing import Any
import numpy as np

from src.config import DEFAULT_KERNEL_SIZE, DEFAULT_SIGMA, SAMPLE_IMAGES_DIR
from src.image_io import load_images_from_folder, save_image
from src.pipeline import process_single_image


def process_images_sequential(
    images: list[tuple[str, np.ndarray]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    """
    Process a list of images sequentially in the current process.

    Iterates through images in the given order, applying the pipeline
    to each image one at a time. Measures total wall-clock execution time.

    Args:
        images: List of (name, rgb_array) pairs.
        params: Filter configuration dictionary with 'kernel_size' and 'sigma'.

    Returns:
        tuple containing:
            - results: List of result dictionaries from process_single_image.
            - total_seconds: Total wall-clock duration in seconds.
    """
    results: list[dict[str, Any]] = []

    # Measure total execution time across the entire batch
    start_total = time.perf_counter()

    for name, array in images:
        task = (name, array, params)
        result = process_single_image(task)
        results.append(result)

    total_seconds = time.perf_counter() - start_total
    return results, total_seconds


def main():
    """Command-line interface for running sequential batch processing."""
    parser = argparse.ArgumentParser(
        description="Run sequential image processing pipeline on a folder of images."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SAMPLE_IMAGES_DIR),
        help="Input folder containing images (default: sample_images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output folder to save processed images (<stem>_gray.png, etc.)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=DEFAULT_KERNEL_SIZE,
        help=f"Gaussian blur kernel size (default: {DEFAULT_KERNEL_SIZE})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Gaussian blur sigma (default: {DEFAULT_SIGMA})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    images = load_images_from_folder(input_path)

    params = {
        "kernel_size": args.kernel_size,
        "sigma": args.sigma,
    }

    # Print summary header
    print(f"Loaded {len(images)} image(s) from '{input_path}'.")
    print(f"Processing sequentially (kernel_size={args.kernel_size}, sigma={args.sigma})...\n")

    # Map name to dimensions (Width x Height) for logging
    dimensions = {name: (arr.shape[1], arr.shape[0]) for name, arr in images}

    # Execute sequential batch processing
    results, total_time = process_images_sequential(images, params)

    # Print individual image timings
    for res in results:
        img_name = res["name"]
        width, height = dimensions[img_name]
        seconds = res["time_seconds"]
        print(f"  {img_name} ({width}x{height}) - {seconds:.4f}s")

    print(f"\nTotal sequential processing time: {total_time:.4f}s")

    # Save output images if an output directory was provided
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            stem = Path(res["name"]).stem
            save_image(res["gray"], output_dir / f"{stem}_gray.png")
            save_image(res["blur"], output_dir / f"{stem}_blur.png")
            save_image(res["edges"], output_dir / f"{stem}_edges.png")
        print(f"Saved processed images to '{output_dir}'.")


if __name__ == "__main__":
    main()
