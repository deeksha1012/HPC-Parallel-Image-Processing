"""
In-memory export utilities for images, figures, tables, and archives.

Provides functions to serialize data directly to bytes for web downloads:
1. image_to_png_bytes: Convert a 2D or 3D NumPy array to PNG bytes.
2. figure_to_png_bytes: Convert a Matplotlib Figure to PNG bytes.
3. dataframe_to_csv_bytes: Convert a pandas DataFrame to UTF-8 CSV bytes.
4. create_results_zip: Package processed image outputs into an in-memory ZIP archive.
"""

import io
from pathlib import Path
from typing import Any, Sequence
import zipfile
import matplotlib.figure
import numpy as np
import pandas as pd
from PIL import Image


def image_to_png_bytes(array: np.ndarray) -> bytes:
    """
    Convert a uint8 NumPy array into in-memory PNG image bytes.

    Args:
        array (np.ndarray): 2D grayscale/edge array or 3D RGB array.

    Returns:
        bytes: Encoded PNG binary data.

    Raises:
        ValueError: If input array is not 2D or 3D.
    """
    if array.ndim == 2:
        img = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        img = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(f"Expected 2D array or 3D (H,W,3) array, got shape {array.shape}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def figure_to_png_bytes(fig: matplotlib.figure.Figure) -> bytes:
    """
    Render and serialize a Matplotlib Figure into in-memory PNG bytes.

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object.

    Returns:
        bytes: Encoded PNG binary data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialize a pandas DataFrame into in-memory CSV bytes with full precision.

    Args:
        df (pd.DataFrame): Data table to serialize.

    Returns:
        bytes: UTF-8 encoded CSV binary data without row indices.
    """
    return df.to_csv(index=False).encode("utf-8")


def create_results_zip(
    results: list[dict[str, Any]],
    which: Sequence[str] = ("gray", "blur", "edges"),
) -> bytes:
    """
    Create an in-memory ZIP archive of processed images organized by filter stage.

    No files are written to disk. Images are saved as:
        <filter_stage>/<stem>_<filter_stage>.png

    Args:
        results: List of dictionaries from process_single_image containing
                 'name', 'gray', 'blur', and 'edges'.
        which: Sequence of filter keys to include (e.g. 'gray', 'blur', 'edges').

    Returns:
        bytes: Complete ZIP archive binary content.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for res in results:
            stem = Path(res["name"]).stem
            for stage in which:
                if stage in res and isinstance(res[stage], np.ndarray):
                    png_bytes = image_to_png_bytes(res[stage])
                    archive_path = f"{stage}/{stem}_{stage}.png"
                    zf.writestr(archive_path, png_bytes)

    return buf.getvalue()
