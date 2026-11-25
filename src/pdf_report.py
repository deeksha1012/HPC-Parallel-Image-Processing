"""
Automated PDF Benchmark Report Generator for HPC Parallel Image Processing.

Generates an academic, publication-quality A4 PDF report in memory containing:
1. Project title, date, time, and system environment specifications.
2. Experiment configuration (images, resolutions, parameters, repeats).
3. Complete sequential and parallel benchmark results table.
4. Best-performing worker configuration highlight.
5. Automated performance analysis and conclusion based on real measurements.
6. Execution time, speedup, and parallel efficiency charts.
7. Visual gallery of all uploaded images across all 4 processing stages
   (Original, Grayscale, Gaussian Blur, Sobel Edges) with preserved aspect ratios.

Uses ReportLab with dynamic two-pass pagination ("Page X of Y").
"""

from datetime import datetime
import io
from pathlib import Path
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.plots import plot_efficiency, plot_execution_time, plot_speedup


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that dynamically calculates the total page count
    and renders consistent running headers and footers across all pages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Running header on subsequent pages
        if self._pageNumber > 1:
            self.drawString(
                40,
                841.89 - 28,
                "High Performance Image Processing — Benchmark & Performance Report",
            )
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(40, 841.89 - 32, 595.27 - 40, 841.89 - 32)

        # Running footer on all pages
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(595.27 - 40, 25, footer_text)
        self.drawString(
            40,
            25,
            "Confidential & Academic — High Performance Computing Mini Project",
        )
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(40, 35, 595.27 - 40, 35)

        self.restoreState()


def _downsample_image_to_bytes(
    img_array: np.ndarray,
    max_dim: int = 400,
) -> tuple[io.BytesIO, float, float]:
    """
    Downsample a NumPy image array in memory while strictly preserving aspect ratio.

    Args:
        img_array: 2D uint8 (grayscale) or 3D uint8 (RGB) array.
        max_dim: Maximum dimension (width or height) in pixels.

    Returns:
        tuple containing (in-memory BytesIO buffer, original width, original height).
    """
    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        pil_img = Image.fromarray(img_array, mode="RGB")
    else:
        # Fallback for unexpected dimensions
        pil_img = Image.fromarray(img_array.astype(np.uint8))

    orig_w, orig_h = pil_img.size

    # Resize if larger than max_dim to keep PDF compact and fast to render
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    # JPEG encoding provides excellent compression for embedded PDF report graphics
    if pil_img.mode == "L":
        pil_img.save(buf, format="JPEG", quality=85)
    else:
        pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    return buf, float(orig_w), float(orig_h)


def _render_matplotlib_chart(fig: plt.Figure, width_pt: float = 510, height_pt: float = 230) -> RLImage:
    """Serialize a Matplotlib Figure into a ReportLab Image flowable and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width_pt, height=height_pt)


def _generate_performance_analysis_text(df: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    """
    Dynamically analyze benchmark measurements to generate summary metrics and conclusions.

    Returns:
        tuple containing (best_config_summary, detailed_analysis_paragraphs, metrics_dict).
    """
    seq_row = df[df["mode"] == "Sequential"].iloc[0]
    seq_time = float(seq_row["mean_time_s"])
    par_df = df[df["mode"] == "Parallel"].copy()

    if par_df.empty:
        summary = "Sequential Processing Only (No parallel configurations tested)"
        analysis = (
            "Only sequential execution was evaluated. To assess parallel speedup and multi-core "
            "efficiency, select one or more worker process counts in the sidebar."
        )
        return summary, analysis, {"best_workers": 1, "best_time": seq_time, "speedup": 1.0, "eff": 1.0}

    # Identify fastest parallel run and peak speedup
    fastest_idx = par_df["mean_time_s"].idxmin()
    fastest_row = par_df.loc[fastest_idx]
    best_workers = int(fastest_row["workers"])
    best_time = float(fastest_row["mean_time_s"])
    best_speedup = float(fastest_row["speedup"])
    best_efficiency = float(fastest_row["efficiency"]) * 100.0

    max_sp_row = par_df.loc[par_df["speedup"].idxmax()]
    max_sp_workers = int(max_sp_row["workers"])
    max_speedup = float(max_sp_row["speedup"])

    metrics_dict = {
        "best_workers": best_workers,
        "best_time": best_time,
        "speedup": best_speedup,
        "eff": best_efficiency,
        "seq_time": seq_time,
    }

    if best_time < seq_time:
        reduction_pct = (1.0 - best_time / seq_time) * 100.0
        summary = (
            f"Optimal Configuration: {best_workers} Worker Processes | "
            f"Execution Time: {best_time:.3f} s | Speedup: {best_speedup:.2f}x | "
            f"Efficiency: {best_efficiency:.1f}%"
        )
        analysis_part1 = (
            f"<b>Parallel Speedup & Scalability:</b> The parallel processing pipeline successfully accelerated "
            f"batch image processing over the single-threaded sequential baseline ({seq_time:.3f} s). "
            f"The best-performing allocation utilized <b>{best_workers} worker processes</b>, reducing "
            f"total wall-clock execution time to <b>{best_time:.3f} s</b>—representing a <b>{reduction_pct:.1f}% time reduction</b> "
            f"and a peak speedup of <b>{best_speedup:.2f}x</b>."
        )
    else:
        summary = (
            f"Baseline Configuration: Sequential Processing ({seq_time:.3f} s) matched or outperformed "
            f"parallel execution due to multiprocessing startup overhead."
        )
        analysis_part1 = (
            f"<b>Overhead vs. Computation Trade-off:</b> On this specific workload, the single-core sequential baseline "
            f"({seq_time:.3f} s) matched or outperformed parallel execution (fastest: {best_time:.3f} s with {best_workers} workers). "
            f"For small image batches or lightweight filtering operations, the fixed overhead of spawning new Python interpreter "
            f"processes (macOS <i>spawn</i> method) and serializing/deserializing NumPy arrays across IPC pipes outweighs "
            f"the computational gains of multi-core parallelism."
        )

    # Discuss efficiency trend and Amdahl's Law
    min_eff_workers = int(par_df.loc[par_df["efficiency"].idxmin()]["workers"])
    min_eff_val = float(par_df["efficiency"].min()) * 100.0

    analysis_part2 = (
        f"<b>Efficiency Degradation & Amdahl's Law:</b> Parallel efficiency reached a minimum of "
        f"<b>{min_eff_val:.1f}%</b> at {min_eff_workers} worker processes. As worker count increases, "
        f"inter-process communication latency, memory bus bandwidth contention, and synchronization overhead "
        f"consume an increasing fraction of runtime. Furthermore, on Apple Silicon architectures, allocating "
        f"workers beyond the physical Performance (P-core) count schedules tasks onto lower-frequency Efficiency "
        f"(E-cores), creating load imbalances where faster cores wait for slower cores."
    )

    analysis_part3 = (
        "<b>Conclusion & Recommendations:</b> Data parallelism across independent image tasks is highly effective "
        "for substantial workloads. To further maximize parallel performance, future optimizations should consider "
        "shared memory (<code>multiprocessing.shared_memory</code>) to eliminate IPC array serialization, spatial tiling "
        "for large individual images, and GPU/Metal acceleration for SIMD operations."
    )

    full_analysis = f"{analysis_part1}<br/><br/>{analysis_part2}<br/><br/>{analysis_part3}"
    return summary, full_analysis, metrics_dict


def generate_benchmark_pdf(
    df: pd.DataFrame,
    last_results: list[dict[str, Any]],
    original_images: list[tuple[str, np.ndarray]],
    params: dict[str, Any],
    system_info: Optional[dict[str, Any]] = None,
) -> bytes:
    """
    Generate a complete, publication-ready A4 PDF report in memory.

    Args:
        df: Benchmark DataFrame with timing, speedup, and efficiency metrics.
        last_results: List of processed image dictionaries ('name', 'gray', 'blur', 'edges').
        original_images: List of (name, rgb_uint8_array) tuples.
        params: Filter configuration dictionary ('kernel_size', 'sigma').
        system_info: Hardware and environment specification dictionary.

    Returns:
        bytes: Raw PDF binary stream ready for download or saving.
    """
    system_info = system_info or {}
    pdf_buffer = io.BytesIO()

    # Document setup: A4 with 40pt (~14mm) margins
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=46,
    )

    # Styles
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2942"),
        alignment=0,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f2942"),
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1b365d"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2d3748"),
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
        spaceBefore=3,
        spaceAfter=8,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=1,  # Center
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
    )

    story = []

    # =========================================================================
    # 1. Title & Header
    # =========================================================================
    story.append(Paragraph("High Performance Image Processing", title_style))
    story.append(Paragraph("Comprehensive Parallel Benchmarking & Performance Analysis Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2942"), spaceAfter=10))

    # Metadata & Hardware Table
    timestamp_str = system_info.get("timestamp", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
    platform_str = system_info.get("platform", "macOS")
    arch_str = system_info.get("machine", "arm64")
    cores_str = str(system_info.get("cpu_count", "N/A"))
    py_ver = system_info.get("python_version", "3.12")
    start_method = system_info.get("start_method", "spawn")

    meta_data = [
        [
            Paragraph("<b>Report Generated:</b>", body_style),
            Paragraph(str(timestamp_str), body_style),
            Paragraph("<b>CPU Core Count:</b>", body_style),
            Paragraph(f"{cores_str} logical cores", body_style),
        ],
        [
            Paragraph("<b>Platform & OS:</b>", body_style),
            Paragraph(f"{platform_str} ({arch_str})", body_style),
            Paragraph("<b>Process Start Method:</b>", body_style),
            Paragraph(str(start_method), body_style),
        ],
        [
            Paragraph("<b>Python Runtime:</b>", body_style),
            Paragraph(f"v{py_ver}", body_style),
            Paragraph("<b>Execution Model:</b>", body_style),
            Paragraph("Data-Parallel Process Pool", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[110, 150, 110, 145])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 2. Experiment Configuration
    # =========================================================================
    story.append(Paragraph("1. Experiment Configuration", h1_style))

    num_images = len(original_images)
    total_pixels = sum(arr.shape[0] * arr.shape[1] for _, arr in original_images)
    total_mp = total_pixels / 1_000_000.0

    # Collect unique image dimensions
    dims_set = {f"{arr.shape[1]}x{arr.shape[0]}" for _, arr in original_images}
    dims_str = ", ".join(sorted(dims_set))

    kernel_size = params.get("kernel_size", 15)
    sigma = params.get("sigma", 3.0)
    repeats = int(df["runs"].iloc[0]) if not df.empty else 1
    worker_list = sorted(df[df["mode"] == "Parallel"]["workers"].astype(int).tolist())
    workers_str = ", ".join(map(str, worker_list)) if worker_list else "None"

    config_data = [
        [
            Paragraph("<b>Input Images:</b>", body_style),
            Paragraph(f"{num_images} image(s)", body_style),
            Paragraph("<b>Gaussian Kernel Size:</b>", body_style),
            Paragraph(f"{kernel_size} x {kernel_size} (odd window)", body_style),
        ],
        [
            Paragraph("<b>Resolutions:</b>", body_style),
            Paragraph(dims_str, body_style),
            Paragraph("<b>Gaussian Sigma (σ):</b>", body_style),
            Paragraph(f"{sigma:.1f}", body_style),
        ],
        [
            Paragraph("<b>Total Workload:</b>", body_style),
            Paragraph(f"{total_mp:.2f} Megapixels", body_style),
            Paragraph("<b>Benchmark Repeats:</b>", body_style),
            Paragraph(f"{repeats} runs per configuration", body_style),
        ],
        [
            Paragraph("<b>Parallel Workers:</b>", body_style),
            Paragraph(f"[{workers_str}] cores", body_style),
            Paragraph("<b>Pipeline Stages:</b>", body_style),
            Paragraph("Grayscale → Blur → Sobel Edges", body_style),
        ],
    ]
    config_table = Table(config_data, colWidths=[110, 150, 110, 145])
    config_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#edf2f7")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(config_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. Benchmark Results Table
    # =========================================================================
    story.append(Paragraph("2. Benchmark Results Table", h1_style))

    table_headers = [
        Paragraph("<b>Mode</b>", table_cell_bold),
        Paragraph("<b>Workers</b>", table_cell_bold),
        Paragraph("<b>Runs</b>", table_cell_bold),
        Paragraph("<b>Mean Time</b>", table_cell_bold),
        Paragraph("<b>Min Time</b>", table_cell_bold),
        Paragraph("<b>Std Dev</b>", table_cell_bold),
        Paragraph("<b>Speedup</b>", table_cell_bold),
        Paragraph("<b>Efficiency</b>", table_cell_bold),
    ]
    table_rows = [table_headers]

    for _, row in df.iterrows():
        mode_val = str(row["mode"])
        workers_val = str(int(row["workers"]))
        runs_val = str(int(row["runs"]))
        mean_s = f"{float(row['mean_time_s']):.3f} s"
        min_s = f"{float(row['min_time_s']):.3f} s"
        std_s = f"{float(row['std_time_s']):.3f} s"
        speedup_str = f"{float(row['speedup']):.2f}x"
        eff_str = f"{float(row['efficiency']) * 100.0:.1f}%"

        table_rows.append(
            [
                Paragraph(mode_val, table_cell_style),
                Paragraph(workers_val, table_cell_style),
                Paragraph(runs_val, table_cell_style),
                Paragraph(mean_s, table_cell_style),
                Paragraph(min_s, table_cell_style),
                Paragraph(std_s, table_cell_style),
                Paragraph(speedup_str, table_cell_style),
                Paragraph(eff_str, table_cell_style),
            ]
        )

    results_table = Table(
        table_rows,
        colWidths=[70, 50, 45, 70, 70, 65, 65, 80],
    )
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2942")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0f2942")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(results_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # 4. Best Configuration & Automated Performance Analysis
    # =========================================================================
    story.append(Paragraph("3. Performance Analysis & Executive Summary", h1_style))
    best_summary, analysis_html, _ = _generate_performance_analysis_text(df)

    # Highlight Callout Box
    callout_data = [[Paragraph(f"<b>KEY FINDING:</b> {best_summary}", body_style)]]
    callout_table = Table(callout_data, colWidths=[515])
    callout_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ebf8ff")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#3182ce")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(callout_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(analysis_html, body_style))
    story.append(Spacer(1, 14))

    # =========================================================================
    # 5. Performance Graphs
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Benchmark Performance Visualizations", h1_style))
    story.append(
        Paragraph(
            "Empirical scaling charts comparing execution time, parallel speedup, and core efficiency. "
            "All charts are rendered dynamically from the current benchmark dataset.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # Chart 1: Execution Time
    fig_time = plot_execution_time(df)
    story.append(KeepTogether([
        Paragraph("<b>Figure 1: Mean Execution Time Comparison</b>", h2_style),
        _render_matplotlib_chart(fig_time, width_pt=515, height_pt=200),
        Paragraph(
            "<i>Note: Lower bars indicate faster processing. Error bars show standard deviation across repeated trials.</i>",
            caption_style,
        ),
    ]))
    story.append(Spacer(1, 6))

    # Chart 2: Speedup
    fig_speedup = plot_speedup(df)
    story.append(KeepTogether([
        Paragraph("<b>Figure 2: Parallel Speedup vs. Worker Process Count</b>", h2_style),
        _render_matplotlib_chart(fig_speedup, width_pt=515, height_pt=200),
        Paragraph(
            "<i>Note: Points above 1.0 demonstrate acceleration. The dashed line depicts theoretical linear scaling (S = p).</i>",
            caption_style,
        ),
    ]))
    story.append(Spacer(1, 6))

    # Chart 3: Efficiency
    fig_eff = plot_efficiency(df)
    story.append(KeepTogether([
        Paragraph("<b>Figure 3: Parallel Efficiency vs. Worker Process Count</b>", h2_style),
        _render_matplotlib_chart(fig_eff, width_pt=515, height_pt=200),
        Paragraph(
            "<i>Note: Dashed line at 100% represents ideal scaling. Downward slope illustrates increasing IPC and synchronization overhead.</i>",
            caption_style,
        ),
    ]))

    # =========================================================================
    # 6. Processed Images Gallery (All Uploaded Images)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Processed Images Visual Gallery", h1_style))
    story.append(
        Paragraph(
            "Visual verification comparing all uploaded original images against their intermediate "
            "and final filtering stages: <b>Original RGB</b>, <b>Grayscale (ITU-R BT.601)</b>, "
            "<b>Separable Gaussian Blur</b>, and <b>3x3 Sobel Edge Detection</b>.",
            body_style,
        )
    )
    story.append(Spacer(1, 10))

    # Map results by image filename
    res_by_name = {res["name"]: res for res in last_results}

    # Column width for each stage: 515 / 4 = ~128 pt
    col_w = 128.0
    max_box_w = 120.0
    max_box_h = 80.0

    for idx, (img_name, orig_arr) in enumerate(original_images, start=1):
        res = res_by_name.get(img_name)
        if not res:
            continue

        orig_h, orig_w = orig_arr.shape[:2]

        # Prepare in-memory downsampled thumbnails maintaining exact aspect ratio
        b_orig, _, _ = _downsample_image_to_bytes(orig_arr, max_dim=350)
        b_gray, _, _ = _downsample_image_to_bytes(res["gray"], max_dim=350)
        b_blur, _, _ = _downsample_image_to_bytes(res["blur"], max_dim=350)
        b_edge, _, _ = _downsample_image_to_bytes(res["edges"], max_dim=350)

        # Calculate display dimensions within max_box_w x max_box_h
        aspect = orig_w / float(orig_h)
        disp_w = max_box_w
        disp_h = disp_w / aspect
        if disp_h > max_box_h:
            disp_h = max_box_h
            disp_w = disp_h * aspect

        cell_orig = [
            RLImage(b_orig, width=disp_w, height=disp_h),
            Paragraph("<b>1. Original RGB</b>", table_cell_style),
        ]
        cell_gray = [
            RLImage(b_gray, width=disp_w, height=disp_h),
            Paragraph("<b>2. Grayscale</b>", table_cell_style),
        ]
        cell_blur = [
            RLImage(b_blur, width=disp_w, height=disp_h),
            Paragraph("<b>3. Gaussian Blur</b>", table_cell_style),
        ]
        cell_edge = [
            RLImage(b_edge, width=disp_w, height=disp_h),
            Paragraph("<b>4. Sobel Edges</b>", table_cell_style),
        ]

        gallery_matrix = [[cell_orig, cell_gray, cell_blur, cell_edge]]
        gallery_table = Table(gallery_matrix, colWidths=[col_w, col_w, col_w, col_w])
        gallery_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ]
            )
        )

        item_elements = [
            Paragraph(f"<b>Image #{idx}: {img_name}</b> ({orig_w} × {orig_h} px)", h2_style),
            gallery_table,
            Spacer(1, 8),
        ]
        story.append(KeepTogether(item_elements))

    # Build the document using the NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
