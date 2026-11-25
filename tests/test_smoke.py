"""
Smoke test suite to verify initial project configuration and environment setup.
"""

from pathlib import Path
from src.config import (
    DEFAULT_KERNEL_SIZE,
    SUPPORTED_EXTENSIONS,
    SAMPLE_IMAGES_DIR,
    RESULTS_DIR,
)


def test_config():
    """
    Verify that basic configuration constants are properly defined and valid.
    """
    # 1. Blur kernel size must be an odd positive integer for symmetric neighborhood filtering
    assert DEFAULT_KERNEL_SIZE % 2 == 1, "DEFAULT_KERNEL_SIZE must be an odd number"

    # 2. Supported extensions must include common formats such as '.png'
    assert ".png" in SUPPORTED_EXTENSIONS, "SUPPORTED_EXTENSIONS must contain '.png'"

    # 3. Path constants must be valid pathlib.Path instances
    assert isinstance(SAMPLE_IMAGES_DIR, Path), "SAMPLE_IMAGES_DIR must be an instance of Path"
    assert isinstance(RESULTS_DIR, Path), "RESULTS_DIR must be an instance of Path"
