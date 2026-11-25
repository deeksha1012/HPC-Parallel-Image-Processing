"""
Modular Streamlit tab renderers for the HPC Image Processing Lab.

Organizes the application interface into 6 dedicated, clean views:
1. Overview: Executive summary cards, workflow guide, and quick metrics.
2. Image Processing: 4-stage gallery, interactive before/after slider, full previews.
3. Performance Analytics: Scaling charts, speedup comparisons, and metrics table.
4. System Monitor: Hardware profiling and Module 10 feature preview.
5. Benchmark History: Historical run management and Module 10 feature preview.
6. Reports: Export center for PDF report, CSV data, ZIP archive, and charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.export import (
    create_results_zip,
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
    image_to_png_bytes,
)
from src.pdf_report import generate_benchmark_pdf
from src.plots import plot_efficiency, plot_execution_time, plot_speedup
from src.ui.components import (
    format_bytes_size,
    render_comparison_slider,
    render_metric_card,
)


def render_overview_tab(
    df: Optional[pd.DataFrame],
    last_results: Optional[List[Dict[str, Any]]],
    loaded_images: List[Tuple[str, np.ndarray]],
    system_info: Dict[str, Any],
    params: Dict[str, Any],
    selected_workers: List[int],
) -> None:
    """Render the Overview tab with executive summary cards and workload statistics."""
    st.markdown("### 📊 Lab Overview & Executive Dashboard")
    st.write(
        "A high-performance computing evaluation platform designed to benchmark pure-NumPy "
        "image filtering pipelines across sequential and multi-core data-parallel architectures."
    )

    # 1. Compute dynamic workload statistics
    num_images = len(loaded_images)
    total_pixels = sum(arr.shape[0] * arr.shape[1] for _, arr in loaded_images) if loaded_images else 0
    total_mp = total_pixels / 1_000_000.0
    cpu_cores = system_info.get("cpu_count", "N/A")
    workers_text = ", ".join(map(str, sorted(selected_workers))) if selected_workers else "None"

    # Compute performance highlights if benchmark data exists
    if df is not None and not df.empty:
        seq_row = df[df["mode"] == "Sequential"].iloc[0]
        seq_time = float(seq_row["mean_time_s"])
        par_df = df[df["mode"] == "Parallel"]

        # Fastest overall run across all modes
        fastest_overall_idx = df["mean_time_s"].idxmin()
        fastest_overall_row = df.loc[fastest_overall_idx]
        fastest_time_str = f"{float(fastest_overall_row['mean_time_s']):.3f} s"
        fastest_mode_str = f"{fastest_overall_row['mode']} ({int(fastest_overall_row['workers'])}w)"

        if not par_df.empty:
            max_sp_row = par_df.loc[par_df["speedup"].idxmax()]
            best_speedup_str = f"{float(max_sp_row['speedup']):.2f}x"
            best_speedup_sub = f"{int(max_sp_row['workers'])} workers"

            fastest_par_row = par_df.loc[par_df["mean_time_s"].idxmin()]
            best_config_str = f"{int(fastest_par_row['workers'])} Workers"
            best_config_sub = f"{float(fastest_par_row['mean_time_s']):.3f}s ({float(fastest_par_row['efficiency']) * 100:.1f}% eff)"
        else:
            best_speedup_str = "1.00x"
            best_speedup_sub = "Sequential only"
            best_config_str = "Sequential"
            best_config_sub = f"{seq_time:.3f} s"
    else:
        fastest_time_str = "Pending run"
        fastest_mode_str = "Execute benchmark"
        best_speedup_str = "Pending run"
        best_speedup_sub = "Execute benchmark"
        best_config_str = "Pending run"
        best_config_sub = "Execute benchmark"

    # 2. Summary cards (2 rows of clean metric cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Loaded Images", f"{num_images}", "Ready for processing", icon="🖼️")
    with c2:
        render_metric_card("Total Workload", f"{total_mp:.2f} MP", f"{total_pixels:,} total pixels", icon="📐")
    with c3:
        render_metric_card("Available Cores", f"{cpu_cores}", f"Architecture: {system_info.get('machine', 'arm64')}", icon="🖥️")
    with c4:
        render_metric_card("Selected Workers", f"[{workers_text}]", "Active allocation list", icon="⚙️")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card("Fastest Time", fastest_time_str, fastest_mode_str, icon="⚡")
    with c6:
        render_metric_card("Peak Speedup", best_speedup_str, best_speedup_sub, icon="🚀")
    with c7:
        render_metric_card("Best Configuration", best_config_str, best_config_sub, icon="🏆")
    with c8:
        repeats_val = int(df["runs"].iloc[0]) if df is not None and not df.empty else params.get("repeats", 3)
        render_metric_card("Benchmark Repeats", f"{repeats_val} Runs", "Per configuration trial", icon="🔁")

    st.markdown("---")

    # 3. Dynamic Status & Guidance
    if df is not None and not df.empty:
        st.markdown(
            f"""
            <div class="hpc-callout-success">
                <b>Benchmark Active:</b> Completed execution across <b>{len(df)} configurations</b>.
                Explore the <b>Image Processing</b> tab for filter comparisons, <b>Performance Analytics</b> for speedup curves,
                or <b>Reports</b> to download the publication-quality PDF report.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif loaded_images:
        st.markdown(
            """
            <div class="hpc-callout-info">
                <b>Images Loaded & Ready:</b> Configure filter parameters (kernel size, sigma) and worker counts in the sidebar,
                then click <b>▶️ Run Benchmark</b> to start sequential and parallel processing.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hpc-callout-warning">
                <b>No Images Loaded:</b> Choose <b>Generate Sample Images</b> in the sidebar for instant synthetic testing,
                or upload your own <code>.png</code> / <code>.jpg</code> files to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Lab Workflow Guide
    with st.expander("📖 HPC Lab Architecture & Workflow Guide", expanded=False):
        st.markdown(
            """
            1. **Sequential Baseline ($T_1$)**: Evaluates a pure-NumPy pipeline (Grayscale $\\to$ Gaussian Blur $\\to$ Sobel Edges)
               executed on a single core after an untimed warm-up pass.
            2. **Data-Parallel Processing ($T_p$)**: Distributes independent images across a pool of worker processes
               using Python's `multiprocessing` library under the macOS-safe `spawn` context.
            3. **Speedup ($S_p$) & Efficiency ($E_p$)**:
               $$\\text{Speedup } S_p = \\frac{T_1}{T_p}, \\quad \\text{Efficiency } E_p = \\frac{S_p}{p}$$
            4. **Amdahl's Law & IPC Friction**: For small image sets, inter-process pickling and process creation
               can exceed computational savings, yielding speedup below 1.0. Larger workloads amortize this overhead.
            """
        )


def render_image_processing_tab(
    loaded_images: List[Tuple[str, np.ndarray]],
    last_results: Optional[List[Dict[str, Any]]],
    params: Dict[str, Any],
) -> None:
    """Render the Image Processing tab with dropdown selection, metadata, interactive slider, and previews."""
    st.markdown("### 🖼️ Interactive Image Processing & Visual Inspection")

    if not loaded_images:
        st.info("👈 Upload images or generate sample images in the sidebar to inspect processing operations.")
        return

    # Image selector dropdown
    image_names = [name for name, _ in loaded_images]
    selected_name = st.selectbox(
        "Select an image to inspect:",
        options=image_names,
        key="ui_inspect_image_select",
    )

    # Locate chosen image array
    orig_dict = {name: arr for name, arr in loaded_images}
    orig_arr = orig_dict[selected_name]
    orig_h, orig_w = orig_arr.shape[:2]
    uncompressed_bytes = orig_arr.nbytes
    mp = (orig_w * orig_h) / 1_000_000.0

    # Display image metadata badges
    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 1rem 0;">
            <span class="hpc-meta-pill">📏 <b>Resolution:</b> {orig_w} × {orig_h} px</span>
            <span class="hpc-meta-pill">📐 <b>Workload:</b> {mp:.2f} Megapixels</span>
            <span class="hpc-meta-pill">💾 <b>Memory (RGB):</b> {format_bytes_size(uncompressed_bytes)}</span>
            <span class="hpc-meta-pill">⚙️ <b>Kernel Size:</b> {params.get('kernel_size', 15)}×{params.get('kernel_size', 15)}</span>
            <span class="hpc-meta-pill">🔍 <b>Sigma:</b> {params.get('sigma', 3.0):.1f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if processed results exist
    res_dict = {res["name"]: res for res in last_results} if last_results else {}
    has_results = selected_name in res_dict

    if not has_results:
        st.info("Run the benchmark to generate intermediate Grayscale, Gaussian Blur, and Sobel Edge outputs.")
        st.image(orig_arr, caption=f"Original RGB: {selected_name}", use_container_width=True)
        return

    res = res_dict[selected_name]
    stem = Path(selected_name).stem

    # 1. Interactive Before-and-After Comparison Slider
    st.markdown("#### ↔️ Interactive Before & After Comparison")
    compare_stage = st.radio(
        "Compare Original image with:",
        options=["Grayscale (ITU-R BT.601)", "Gaussian Blur (Separable 1D)", "Sobel Edges (3×3 Gradient)"],
        horizontal=True,
        key="compare_stage_radio",
    )

    if "Grayscale" in compare_stage:
        target_arr = res["gray"]
        target_label = "Grayscale Output"
    elif "Gaussian" in compare_stage:
        target_arr = res["blur"]
        target_label = "Gaussian Blurred Output"
    else:
        target_arr = res["edges"]
        target_label = "Sobel Edge Detection Output"

    view_mode = st.radio(
        "Comparison Mode:",
        options=["Interactive Drag Slider", "Side-by-Side Dual View"],
        horizontal=True,
        key="compare_view_mode",
    )

    if view_mode == "Interactive Drag Slider":
        st.caption("Drag the vertical divider handle left or right to reveal processing transformations:")
        render_comparison_slider(
            img_before=orig_arr,
            img_after=target_arr,
            label_before="Original (RGB)",
            label_after=target_label,
            height=430,
        )
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(orig_arr, caption="Original RGB", use_container_width=True)
        with col_right:
            st.image(target_arr, caption=target_label, use_container_width=True)

    st.markdown("---")

    # 2. 4-Stage Visual Processing Pipeline Gallery
    st.markdown("#### 🔄 Complete 4-Stage Processing Pipeline")
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.image(orig_arr, caption="1. Original RGB", use_container_width=True)
        st.download_button(
            "⬇️ Original PNG",
            data=image_to_png_bytes(orig_arr),
            file_name=f"{stem}_original.png",
            mime="image/png",
            key=f"dl_tab_orig_{stem}",
            use_container_width=True,
        )

    with g2:
        st.image(res["gray"], caption="2. Grayscale (BT.601)", use_container_width=True)
        st.download_button(
            "⬇️ Grayscale PNG",
            data=image_to_png_bytes(res["gray"]),
            file_name=f"{stem}_gray.png",
            mime="image/png",
            key=f"dl_tab_gray_{stem}",
            use_container_width=True,
        )

    with g3:
        st.image(res["blur"], caption="3. Gaussian Blur", use_container_width=True)
        st.download_button(
            "⬇️ Blur PNG",
            data=image_to_png_bytes(res["blur"]),
            file_name=f"{stem}_blur.png",
            mime="image/png",
            key=f"dl_tab_blur_{stem}",
            use_container_width=True,
        )

    with g4:
        st.image(res["edges"], caption="4. Sobel Edges", use_container_width=True)
        st.download_button(
            "⬇️ Edges PNG",
            data=image_to_png_bytes(res["edges"]),
            file_name=f"{stem}_edges.png",
            mime="image/png",
            key=f"dl_tab_edges_{stem}",
            use_container_width=True,
        )

    # 3. Full-Size Expanders for In-Depth Examination
    with st.expander("🔍 View Full-Resolution Images for Detail Inspection"):
        st.write("**Original RGB Image**")
        st.image(orig_arr, use_container_width=True)
        st.write("**Grayscale Image (ITU-R BT.601 Luminance)**")
        st.image(res["gray"], use_container_width=True)
        st.write("**Gaussian Blurred Image (Noise Attenuation)**")
        st.image(res["blur"], use_container_width=True)
        st.write("**Sobel Edge Intensity Map (Normalized Gradient)**")
        st.image(res["edges"], use_container_width=True)


def render_analytics_tab(df: Optional[pd.DataFrame]) -> None:
    """Render the Performance Analytics tab with scaling charts, highlights, and tabular metrics."""
    st.markdown("### ⚡ HPC Scaling & Performance Analytics")

    if df is None or df.empty:
        st.info("👈 Run the benchmark from the sidebar to generate execution time curves, speedup charts, and efficiency plots.")
        return

    seq_row = df[df["mode"] == "Sequential"].iloc[0]
    seq_time = float(seq_row["mean_time_s"])
    par_df = df[df["mode"] == "Parallel"]

    # Highlights separating fastest overall from best parallel configuration
    st.markdown("#### 🏆 Performance Highlights")
    fastest_overall_idx = df["mean_time_s"].idxmin()
    fastest_overall_row = df.loc[fastest_overall_idx]

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown(
            f"""
            <div class="hpc-card" style="border-left: 4px solid #3b82f6;">
                <div class="hpc-card-title">🌟 Fastest Overall Configuration</div>
                <div class="hpc-card-value">{float(fastest_overall_row['mean_time_s']):.3f} s</div>
                <div class="hpc-card-subvalue"><b>Mode:</b> {fastest_overall_row['mode']} | <b>Cores:</b> {int(fastest_overall_row['workers'])} worker(s)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h_col2:
        if not par_df.empty:
            best_sp_row = par_df.loc[par_df["speedup"].idxmax()]
            st.markdown(
                f"""
                <div class="hpc-card" style="border-left: 4px solid #10b981;">
                    <div class="hpc-card-title">🚀 Best Parallel Configuration</div>
                    <div class="hpc-card-value">{float(best_sp_row['speedup']):.2f}x Speedup</div>
                    <div class="hpc-card-subvalue"><b>Workers:</b> {int(best_sp_row['workers'])} cores | <b>Efficiency:</b> {float(best_sp_row['efficiency']) * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="hpc-card">
                    <div class="hpc-card-title">🚀 Best Parallel Configuration</div>
                    <div class="hpc-card-value">N/A</div>
                    <div class="hpc-card-subvalue">No parallel worker counts tested</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Performance Charts
    st.markdown("#### 📈 Scaling Visualizations")
    tab_time, tab_speedup, tab_eff = st.tabs(["Execution Time", "Speedup Curve", "Parallel Efficiency"])

    with tab_time:
        fig_time = plot_execution_time(df)
        st.pyplot(fig_time)
        plt.close(fig_time)
        st.caption(
            "**Execution Time Analysis:** Compares mean execution duration across sequential baseline and parallel core counts. "
            "Error bars represent standard deviation across repeated runs, quantifying stability."
        )

    with tab_speedup:
        fig_speedup = plot_speedup(df)
        st.pyplot(fig_speedup)
        plt.close(fig_speedup)
        st.caption(
            "**Speedup Analysis ($S_p = T_1 / T_p$):** Values above 1.0 indicate acceleration over sequential execution. "
            "The dashed line represents theoretical ideal linear speedup where performance scales proportionally with worker cores."
        )

    with tab_eff:
        fig_eff = plot_efficiency(df)
        st.pyplot(fig_eff)
        plt.close(fig_eff)
        st.caption(
            "**Parallel Efficiency Analysis ($E_p = S_p / p$):** Measures the percentage of core compute power effectively utilized. "
            "The 100% line represents ideal scaling. The downward slope illustrates increasing IPC serialization and scheduling overhead."
        )

    st.markdown("---")

    # Complete Metrics Table
    st.markdown("#### 📋 Complete Benchmark Metrics Table")
    display_df = df.copy()
    display_df = display_df.rename(
        columns={
            "mode": "Mode",
            "workers": "Workers",
            "runs": "Runs",
            "mean_time_s": "Mean Time (s)",
            "min_time_s": "Min Time (s)",
            "std_time_s": "Std Dev (s)",
            "speedup": "Speedup",
            "efficiency": "Efficiency",
            "num_images": "Images",
            "total_megapixels": "Megapixels",
            "run_times_s": "Individual Runs (s)",
        }
    )

    display_df["Mean Time (s)"] = display_df["Mean Time (s)"].map(lambda x: f"{x:.3f}")
    display_df["Min Time (s)"] = display_df["Min Time (s)"].map(lambda x: f"{x:.3f}")
    display_df["Std Dev (s)"] = display_df["Std Dev (s)"].map(lambda x: f"{x:.3f}")
    display_df["Speedup"] = display_df["Speedup"].map(lambda x: f"{x:.2f}x")
    display_df["Efficiency"] = display_df["Efficiency"].map(lambda x: f"{x * 100:.1f}%")

    st.dataframe(display_df, use_container_width=True)


def render_system_monitor_tab(system_info: Dict[str, Any]) -> None:
    """Render the System Monitor tab with hardware specs and Module 10 feature preview."""
    st.markdown("### 🖥️ System Monitor & Hardware Profiling")

    st.info(
        "ℹ️ **Feature Preview:** Live CPU core utilization, dynamic memory telemetry, and real-time thermals "
        "are scheduled for implementation in **Module 10**. Below is the static hardware and execution profile for this machine."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("CPU Core Count", f"{system_info.get('cpu_count', 'N/A')}", "Logical execution cores", icon="🧠")
    with m2:
        render_metric_card("Architecture", f"{system_info.get('machine', 'arm64')}", "Apple Silicon native", icon="🏗️")
    with m3:
        render_metric_card("Python Runtime", f"v{system_info.get('python_version', '3.12')}", "CPython interpreter", icon="🐍")
    with m4:
        render_metric_card("Start Method", f"{system_info.get('start_method', 'spawn')}", "macOS thread-safe", icon="🚀")

    st.markdown("#### 🔬 Architecture & Resource Dynamics")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="hpc-card">
                <div class="hpc-card-title">🍎 Apple Silicon Core Asymmetry</div>
                <div style="font-size: 0.9rem; line-height: 1.4; opacity: 0.9;">
                    Apple M-series processors combine <b>Performance (P-cores)</b> and <b>Efficiency (E-cores)</b>.
                    When parallel worker allocations exceed the P-core count, macOS dispatches tasks onto E-cores
                    operating at lower frequencies, introducing scheduling desynchronization.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="hpc-card">
                <div class="hpc-card-title">📦 Inter-Process Communication (IPC)</div>
                <div style="font-size: 0.9rem; line-height: 1.4; opacity: 0.9;">
                    Under the <code>spawn</code> method, each worker is a brand-new Python process.
                    Passing raw image buffers requires pickling NumPy arrays into byte streams across OS pipes,
                    consuming CPU memory bandwidth before filtering computation begins.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_tab(df: Optional[pd.DataFrame]) -> None:
    """Render the Benchmark History tab with session archive and Module 10 feature preview."""
    st.markdown("### 📜 Benchmark History & Experiment Archive")

    st.info(
        "ℹ️ **Feature Preview:** Persistent historical experiment tracking, SQLite/JSON session archives, "
        "and multi-run trend comparisons are scheduled for implementation in **Module 10**."
    )

    if df is not None and not df.empty:
        st.markdown("#### 🕒 Current Session Benchmark Run")
        st.write("The current session's active benchmark data is preserved in memory:")
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No active benchmark data in the current session. Run a benchmark to view session records.")

    # Check if a previous benchmark file exists on disk
    saved_csv = Path("results/benchmark_results.csv")
    if saved_csv.exists():
        with st.expander("📂 View Previous Saved Benchmark on Disk (results/benchmark_results.csv)"):
            disk_df = pd.read_csv(saved_csv)
            st.dataframe(disk_df, use_container_width=True)


def render_reports_tab(
    df: Optional[pd.DataFrame],
    last_results: Optional[List[Dict[str, Any]]],
    loaded_images: List[Tuple[str, np.ndarray]],
    params: Dict[str, Any],
    system_info: Dict[str, Any],
) -> None:
    """Render the Reports tab with organized downloads for PDF, CSV, ZIP, charts, and images."""
    st.markdown("### 📑 Reports & Data Export Center")

    if df is None or df.empty or not last_results:
        st.info("👈 Complete a benchmark execution from the sidebar to generate downloadable PDF reports and archives.")
        return

    # 1. Complete Benchmark Report (PDF)
    st.markdown("#### 📄 Complete Benchmark Report (PDF)")
    st.write(
        "Generate a publication-quality, A4-formatted PDF report containing executive summaries, "
        "experiment configuration, complete tabular metrics, scaling graphs, automated performance analysis, "
        "and the full 4-stage processed image gallery for all uploaded images."
    )

    pdf_bytes = generate_benchmark_pdf(
        df=df,
        last_results=last_results,
        original_images=loaded_images,
        params=params,
        system_info=system_info,
    )
    st.download_button(
        "📥 Download Complete Benchmark Report (PDF)",
        data=pdf_bytes,
        file_name="hpc_benchmark_report.pdf",
        mime="application/pdf",
        key="dl_tab_reports_pdf",
        type="primary",
    )

    st.markdown("---")

    # 2. Raw Benchmark CSV Data Table
    st.markdown("#### 📊 Benchmark Metrics Table (CSV)")
    st.write("Export full-precision benchmark timings, speedup factors, and parallel efficiency metrics in standard CSV format.")
    csv_bytes = dataframe_to_csv_bytes(df)
    st.download_button(
        "⬇️ Download Benchmark Data (CSV)",
        data=csv_bytes,
        file_name="benchmark_results.csv",
        mime="text/csv",
        key="dl_tab_reports_csv",
    )

    st.markdown("---")

    # 3. Batch ZIP Archive of Processed Images
    st.markdown("#### 📦 Processed Images Archive (ZIP)")
    st.write("Download an in-memory ZIP archive containing all processed images organized into folders by filter stage.")
    zip_stages = st.multiselect(
        "Select filter stages to bundle in the ZIP archive:",
        options=["gray", "blur", "edges"],
        default=["gray", "blur", "edges"],
        key="dl_tab_reports_zip_stages",
    )
    if zip_stages:
        zip_bytes = create_results_zip(last_results, which=zip_stages)
        st.download_button(
            "⬇️ Download Processed Images (ZIP Archive)",
            data=zip_bytes,
            file_name="processed_images.zip",
            mime="application/zip",
            key="dl_tab_reports_zip",
        )

    st.markdown("---")

    # 4. High-Resolution Performance Charts (PNG)
    st.markdown("#### 🖼️ High-Resolution Performance Charts (PNG)")
    st.write("Export publication-ready Matplotlib scaling graphs rendered at 150 DPI for presentation slides and papers.")
    c_p1, c_p2, c_p3 = st.columns(3)

    f_time = plot_execution_time(df)
    b_time = figure_to_png_bytes(f_time)
    plt.close(f_time)
    c_p1.download_button(
        "⬇️ Execution Time Graph",
        data=b_time,
        file_name="execution_time.png",
        mime="image/png",
        key="dl_tab_reports_plot_time",
        use_container_width=True,
    )

    f_sp = plot_speedup(df)
    b_sp = figure_to_png_bytes(f_sp)
    plt.close(f_sp)
    c_p2.download_button(
        "⬇️ Speedup Curve Graph",
        data=b_sp,
        file_name="speedup.png",
        mime="image/png",
        key="dl_tab_reports_plot_speedup",
        use_container_width=True,
    )

    f_eff = plot_efficiency(df)
    b_eff = figure_to_png_bytes(f_eff)
    plt.close(f_eff)
    c_p3.download_button(
        "⬇️ Efficiency Curve Graph",
        data=b_eff,
        file_name="efficiency.png",
        mime="image/png",
        key="dl_tab_reports_plot_eff",
        use_container_width=True,
    )
