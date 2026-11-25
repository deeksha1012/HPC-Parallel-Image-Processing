"""
Reusable UI components, custom styling, and interactive visual widgets for Streamlit.
"""

import base64
import io
from typing import Optional
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components


def inject_custom_css() -> None:
    """
    Inject clean, theme-aware CSS styling for modern card surfaces,
    accent borders, badges, and responsive typography in both light and dark themes.
    """
    st.markdown(
        """
        <style>
        /* Modern laboratory styling supporting both light and dark themes */
        .hpc-main-header {
            margin-bottom: 0.5rem;
        }
        .hpc-title {
            font-size: 2.1rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            color: var(--text-color);
            margin-bottom: 0.2rem;
        }
        .hpc-subtitle {
            font-size: 1.05rem;
            font-weight: 450;
            color: var(--text-color);
            opacity: 0.85;
            margin-bottom: 1rem;
        }
        .hpc-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 9999px;
            background: rgba(59, 130, 246, 0.15);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.3);
            margin-bottom: 0.5rem;
        }
        .hpc-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .hpc-card:hover {
            border-color: rgba(59, 130, 246, 0.4);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .hpc-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.75;
            margin-bottom: 0.35rem;
        }
        .hpc-card-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text-color);
            line-height: 1.2;
        }
        .hpc-card-subvalue {
            font-size: 0.8rem;
            opacity: 0.75;
            margin-top: 0.35rem;
        }
        .hpc-callout-success {
            background: rgba(16, 185, 129, 0.08);
            border-left: 4px solid #10b981;
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin: 0.8rem 0;
        }
        .hpc-callout-info {
            background: rgba(59, 130, 246, 0.08);
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin: 0.8rem 0;
        }
        .hpc-callout-warning {
            background: rgba(245, 158, 11, 0.08);
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin: 0.8rem 0;
        }
        .hpc-meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def array_to_base64_jpeg(array: np.ndarray, max_dim: int = 800) -> str:
    """
    Convert a NumPy image array to a compact base64 JPEG string for embedded HTML components.

    Args:
        array: 2D uint8 (grayscale) or 3D uint8 (RGB) NumPy array.
        max_dim: Max image width/height in pixels.

    Returns:
        str: Base64 encoded string.
    """
    if array.ndim == 2:
        img = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        img = Image.fromarray(array, mode="RGB")
    else:
        img = Image.fromarray(array.astype(np.uint8))

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def format_bytes_size(num_bytes: int | float) -> str:
    """Format byte count into human-readable string (KB, MB)."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KB"
    else:
        return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def render_metric_card(
    title: str,
    value: str,
    subvalue: str = "",
    icon: str = "",
) -> None:
    """Render a clean, modern metric card with title, primary value, and subtitle."""
    icon_html = f"{icon} " if icon else ""
    sub_html = f'<div class="hpc-card-subvalue">{subvalue}</div>' if subvalue else ""
    st.markdown(
        f"""
        <div class="hpc-card">
            <div class="hpc-card-title">{icon_html}{title}</div>
            <div class="hpc-card-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_slider(
    img_before: np.ndarray,
    img_after: np.ndarray,
    label_before: str = "Original (RGB)",
    label_after: str = "Processed",
    height: int = 420,
) -> None:
    """
    Render an interactive before-and-after comparison slider where users drag
    a divider horizontally to inspect image processing differences in real time.

    Works 100% offline using an embedded HTML5/CSS clip-path slider.
    """
    b64_before = array_to_base64_jpeg(img_before, max_dim=850)
    b64_after = array_to_base64_jpeg(img_after, max_dim=850)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    body {{
        background: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }}
    .compare-container {{
        position: relative;
        width: 100%;
        max-width: 680px;
        height: {height - 40}px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        background: #000;
        user-select: none;
        -webkit-user-select: none;
    }}
    .img-layer {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }}
    .overlay-layer {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);
    }}
    .overlay-layer img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }}
    .divider-line {{
        position: absolute;
        top: 0;
        bottom: 0;
        left: 50%;
        width: 3px;
        background: #3b82f6;
        transform: translateX(-50%);
        pointer-events: none;
        box-shadow: 0 0 8px rgba(59, 130, 246, 0.8);
    }}
    .divider-handle {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 32px;
        height: 32px;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        pointer-events: none;
        border: 2px solid #ffffff;
    }}
    .range-input {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: ew-resize;
        margin: 0;
        padding: 0;
        z-index: 20;
    }}
    .pill-label {{
        position: absolute;
        bottom: 12px;
        background: rgba(15, 23, 42, 0.75);
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.02em;
        pointer-events: none;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        z-index: 5;
    }}
    .pill-left {{
        left: 12px;
    }}
    .pill-right {{
        right: 12px;
    }}
    </style>
    </head>
    <body>
    <div class="compare-container">
        <!-- Base Layer: Processed Image (Right Side) -->
        <img class="img-layer" src="data:image/jpeg;base64,{b64_after}" />

        <!-- Overlay Layer: Original Image (Left Side clipped) -->
        <div class="overlay-layer" id="overlay">
            <img src="data:image/jpeg;base64,{b64_before}" />
        </div>

        <!-- Vertical Divider Line & Knob -->
        <div class="divider-line" id="divider">
            <div class="divider-handle">&#x2194;</div>
        </div>

        <!-- Interactive Range Slider -->
        <input
            type="range"
            min="0"
            max="100"
            value="50"
            class="range-input"
            id="rangeSlider"
            oninput="handleSlide(this.value)"
        />

        <!-- Badges -->
        <div class="pill-label pill-left">{label_before}</div>
        <div class="pill-label pill-right">{label_after}</div>
    </div>

    <script>
    function handleSlide(val) {{
        document.getElementById('overlay').style.clipPath = 'polygon(0 0, ' + val + '% 0, ' + val + '% 100%, 0 100%)';
        document.getElementById('divider').style.left = val + '%';
    }}
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
