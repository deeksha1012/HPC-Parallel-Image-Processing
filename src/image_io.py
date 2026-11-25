"""
Image input/output helper utilities.

Provides functions to load images from disk or file-like objects (such as
Streamlit uploads), list supported image files in a folder, load batches of
images, and save processed uint8 arrays to disk.
"""

from pathlib import Path
from typing import BinaryIO, Union
import numpy as np
from PIL import Image

from src.config import SUPPORTED_EXTENSIONS


def load_image(source: Union[str, Path, BinaryIO]) -> np.ndarray:
    """
    Load an image from a file path or file-like object and convert it to RGB.

    Converting to RGB ensures consistency:
    - RGBA images (e.g., transparent PNGs) have their alpha channel removed.
    - Grayscale images are expanded to 3 identical color channels.

    Args:
        source (str, Path, or file-like): File system path or binary buffer.

    Returns:
        np.ndarray: 3D uint8 array with shape (Height, Width, 3).
    """
    with Image.open(source) as img:
        # Convert image to 8-bit standard RGB format
        rgb_img = img.convert("RGB")
        return np.array(rgb_img, dtype=np.uint8)


def list_image_files(folder: Union[str, Path]) -> list[Path]:
    """
    Find all supported image files in a folder, sorted alphabetically by filename.

    Args:
        folder (str or Path): Target directory to search.

    Returns:
        list[Path]: Sorted list of Path objects for supported image files.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    supported = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
    image_files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ]

    return sorted(image_files, key=lambda p: p.name)


def load_images_from_folder(folder: Union[str, Path]) -> list[tuple[str, np.ndarray]]:
    """
    Load all supported images from a folder into memory.

    Args:
        folder (str or Path): Target directory containing input images.

    Returns:
        list[tuple[str, np.ndarray]]: List of (filename, RGB uint8 array) pairs.

    Raises:
        ValueError: If no supported images are found in the directory.
    """
    files = list_image_files(folder)
    if not files:
        raise ValueError(
            f"No supported images found in '{folder}'. "
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )

    loaded = []
    for f in files:
        rgb_array = load_image(f)
        loaded.append((f.name, rgb_array))

    return loaded


def save_image(array: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a 2D or 3D uint8 NumPy array to disk as an image file.

    Automatically creates any missing parent directories before saving.

    Args:
        array (np.ndarray): 2D array (grayscale/edges) or 3D array (RGB), uint8.
        path (str or Path): Output file path (e.g., 'results/output.png').

    Raises:
        ValueError: If the input array has an unsupported shape or dimensions.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        # Grayscale or edge intensity map
        img = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        # Standard RGB color image
        img = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(
            f"Expected a 2D array (H, W) or 3D array (H, W, 3), got shape {array.shape}"
        )

    img.save(out_path)
