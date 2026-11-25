"""
Configuration settings and constants for the HPC Image Processing project.
This module defines filesystem paths, default processing parameters, and
application limits used across sequential and parallel processing runs.
"""

from pathlib import Path

# Root directory of the project (one level above src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory where synthetic sample images and input test datasets are stored
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

# Directory where processed output images and benchmark results are saved
RESULTS_DIR = PROJECT_ROOT / "results"

# Supported image file formats for batch processing
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Size of the Gaussian blur kernel window in pixels (must be an odd integer, e.g., 3, 5, 15)
DEFAULT_KERNEL_SIZE = 15

# Standard deviation for the Gaussian blur filter kernel (controls blur intensity)
DEFAULT_SIGMA = 3.0

# Number of times each processing benchmark is repeated to compute a stable average time
DEFAULT_REPEATS = 3

# Maximum number of images allowed to be uploaded at once in the Streamlit web dashboard
MAX_UPLOAD_IMAGES = 30
