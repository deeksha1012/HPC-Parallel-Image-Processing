"""
Data-parallel batch image processor using Python multiprocessing.

Distributes independent image processing tasks across multiple CPU cores
using a worker pool. Each worker process executes the image processing pipeline
(grayscale, Gaussian blur, Sobel edges) on a subset of the images.
"""

import argparse
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any
import numpy as np

from src.config import DEFAULT_KERNEL_SIZE, DEFAULT_SIGMA, SAMPLE_IMAGES_DIR
from src.image_io import load_images_from_folder, save_image
from src.pipeline import process_single_image
from src.sequential import process_images_sequential


def process_images_parallel(
    images: list[tuple[str, np.ndarray]],
    params: dict[str, Any],
    num_workers: int,
    chunksize: int = 1,
) -> tuple[list[dict[str, Any]], float]:
    """
    Process multiple images in parallel using a pool of worker processes.

    Images are mapped across worker processes using the 'spawn' start method,
    ensuring robust behavior across platforms including macOS Apple Silicon.

    Args:
        images: List of (name, rgb_array) pairs.
        params: Filter settings dictionary with 'kernel_size' and 'sigma'.
        num_workers: Number of worker processes to spawn (between 1 and os.cpu_count()).
        chunksize: Number of tasks submitted per worker IPC chunk (default: 1).

    Returns:
        tuple containing:
            - results: List of result dictionaries from process_single_image,
                       guaranteed to be in the same order as input images.
            - total_seconds: Total end-to-end wall-clock time in seconds.

    Raises:
        ValueError: If images is empty or num_workers is out of valid range [1, cpu_count].
        RuntimeError: If an error occurs inside a worker process.
    """
    # 1. Validation checks
    if not images:
        raise ValueError("The images list cannot be empty.")

    max_cpus = os.cpu_count() or 1
    if not isinstance(num_workers, int) or num_workers < 1 or num_workers > max_cpus:
        raise ValueError(
            f"num_workers must be an integer between 1 and {max_cpus}, got {num_workers}."
        )

    # 2. Prepare independent tasks: each image is self-contained with its parameters
    tasks = [(name, array, params) for name, array in images]

    # 3. Create process context using 'spawn'
    # 'spawn' starts fresh Python processes, avoiding macOS fork safety issues
    ctx = multiprocessing.get_context("spawn")

    # Deliberate timing design:
    # We measure time.perf_counter() around the ENTIRE pool context block.
    # This captures the honest end-to-end wall-clock time a user experiences,
    # including process spawning, IPC serialization overhead, processing, and cleanup.
    start_time = time.perf_counter()

    try:
        with ctx.Pool(processes=num_workers) as pool:
            # pool.map preserves the original task ordering in results
            results = pool.map(process_single_image, tasks, chunksize=chunksize)
    except Exception as exc:
        raise RuntimeError(
            f"Parallel processing failed during worker task execution: {exc}"
        ) from exc

    total_seconds = time.perf_counter() - start_time
    return results, total_seconds


def main():
    """Command-line interface for parallel image processing and verification."""
    cpu_limit = os.cpu_count() or 1

    parser = argparse.ArgumentParser(
        description="Run parallel image processing pipeline using Python multiprocessing."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SAMPLE_IMAGES_DIR),
        help="Input folder containing images (default: sample_images)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_limit,
        help=f"Number of worker processes to spawn (1 to {cpu_limit}, default: {cpu_limit})",
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output folder to save processed images (<stem>_gray.png, etc.)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run sequential baseline and verify that parallel outputs match exactly",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    images = load_images_from_folder(input_path)

    params = {
        "kernel_size": args.kernel_size,
        "sigma": args.sigma,
    }

    print(f"Loaded {len(images)} image(s) from '{input_path}'.")
    print(
        f"Processing in parallel with {args.workers} worker(s) "
        f"(kernel_size={args.kernel_size}, sigma={args.sigma})...\n"
    )

    # Execute parallel batch processing
    results, parallel_seconds = process_images_parallel(
        images=images,
        params=params,
        num_workers=args.workers,
    )

    # Inspect distinct worker PIDs that handled tasks
    distinct_pids = {res["pid"] for res in results}

    # Print required normal run summary
    print(f"Images processed: {len(images)}")
    print(f"Workers requested: {args.workers}")
    print(f"Different worker PIDs used: {len(distinct_pids)}")
    print(f"Total parallel processing time: {parallel_seconds:.4f}s")

    # Optional verification against sequential execution
    if args.verify:
        print("\n--- Verification against sequential execution ---")
        seq_results, seq_seconds = process_images_sequential(images, params)
        print(f"Sequential time: {seq_seconds:.4f}s")
        print(f"Parallel time:   {parallel_seconds:.4f}s")

        # Compare outputs image-by-image
        all_identical = True
        differing_images = []

        for par_res, seq_res in zip(results, seq_results):
            img_name = par_res["name"]
            gray_match = np.array_equal(par_res["gray"], seq_res["gray"])
            blur_match = np.array_equal(par_res["blur"], seq_res["blur"])
            edges_match = np.array_equal(par_res["edges"], seq_res["edges"])

            if not (gray_match and blur_match and edges_match):
                all_identical = False
                differing_images.append(img_name)

        if all_identical:
            print("VERIFY: outputs identical")
        else:
            print(f"VERIFY: outputs differ for: {', '.join(differing_images)}")

    # Save output images if an output directory was provided
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            stem = Path(res["name"]).stem
            save_image(res["gray"], output_dir / f"{stem}_gray.png")
            save_image(res["blur"], output_dir / f"{stem}_blur.png")
            save_image(res["edges"], output_dir / f"{stem}_edges.png")
        print(f"\nSaved processed images to '{output_dir}'.")


if __name__ == "__main__":
    main()
