"""
Modular Streamlit tab renderers for the HPC Image Processing Lab.

Organizes the application interface into 7 dedicated, clean views:
1. Overview: Executive summary cards, workload guide, and performance intelligence highlights.
2. Image Processing: 4-stage gallery, interactive before/after comparison slider, and full-resolution inspection.
3. Performance Analytics: 3-way comparative scaling charts, speedup curves, throughput metrics, worker PID distribution, and intelligence recommendations.
4. Scalability & Crossover: Workload scalability testing suite, memory safety estimator, and parallel performance crossover detection.
5. System Monitor: Cross-platform hardware profiling, real-time CPU/RAM gauges, post-benchmark telemetry charts, and architectural insights.
6. Benchmark History: SQLite persistent experiment archive, filtering, multi-run trend charts, experiment inspection, and CSV export.
7. Reports: Export center for comprehensive PDF report, CSV data, ZIP image archive, and Matplotlib graphs.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import streamlit as st

from src.export import (
    create_results_zip,
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
    image_to_png_bytes,
)
from src.history import (
    clear_all_experiments,
    delete_experiment,
    export_history_to_csv_bytes,
    get_all_experiments,
    get_experiment_details,
)
from src.intelligence import compute_throughput_metrics
from src.monitor import (
    aggregate_worker_distribution,
    plot_cpu_utilization,
    plot_memory_utilization,
    plot_worker_task_distribution,
)
from src.pdf_report import generate_benchmark_pdf
from src.plots import plot_efficiency, plot_execution_time, plot_speedup
from src.plots_optimized import (
    plot_efficiency_comparison,
    plot_scalability_crossover,
    plot_speedup_comparison,
    plot_three_way_execution_time,
    plot_throughput_comparison,
)
from src.scalability import (
    estimate_memory_requirements,
    find_crossover_point,
    run_scalability_experiment,
)
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
    intelligence: Optional[Dict[str, Any]] = None,
    monitor_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Overview tab with executive summary cards, workload statistics, and intelligence highlights."""
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
        par_modes = ["Parallel", "Optimized Parallel", "Original Parallel"]
        par_df = df[df["mode"].isin(par_modes)]

        # Fastest overall run across all modes
        fastest_overall_idx = df["mean_time_s"].idxmin()
        fastest_overall_row = df.loc[fastest_overall_idx]
        fastest_time_str = f"{float(fastest_overall_row['mean_time_s']):.3f} s"
        fastest_mode_str = f"{fastest_overall_row['mode']} ({int(fastest_overall_row['workers'])}w)"

        if not par_df.empty:
            max_sp_row = par_df.loc[par_df["speedup"].idxmax()]
            best_speedup_str = f"{float(max_sp_row['speedup']):.2f}x"
            best_speedup_sub = f"{int(max_sp_row['workers'])} workers ({max_sp_row['mode']})"

            fastest_par_row = par_df.loc[par_df["mean_time_s"].idxmin()]
            best_config_str = f"{int(fastest_par_row['workers'])} Workers"
            best_config_sub = f"{float(fastest_par_row['mean_time_s']):.3f}s ({float(fastest_par_row['efficiency']) * 100:.1f}% eff)"
        else:
            best_speedup_str = "1.00x"
            best_speedup_sub = "Sequential only"
            best_config_str = "Sequential"
            best_config_sub = f"{seq_time:.3f} s"

        # Check optimization gain
        has_opt = "Optimized Parallel" in df["mode"].values
        if has_opt:
            opt_rows = df[df["mode"] == "Optimized Parallel"]
            max_gain = float(opt_rows["optimization_gain"].max()) if not opt_rows.empty else 1.0
            gain_label = f"{max_gain:.2f}x Gain"
            gain_sub = "Optimized vs. Original Parallel"
        else:
            gain_label = "Standard"
            gain_sub = "No optimizations enabled"
    else:
        fastest_time_str = "Pending run"
        fastest_mode_str = "Execute benchmark"
        best_speedup_str = "Pending run"
        best_speedup_sub = "Execute benchmark"
        best_config_str = "Pending run"
        best_config_sub = "Execute benchmark"
        gain_label = "Pending run"
        gain_sub = "Execute benchmark"

    # 2. Summary cards (2 rows of clean metric cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Loaded Images", f"{num_images}", "Ready for processing", icon="🖼️")
    with c2:
        render_metric_card("Total Workload", f"{total_mp:.2f} MP", f"{total_pixels:,} total pixels", icon="📐")
    with c3:
        proc_str = system_info.get("processor", system_info.get("machine", "arm64"))
        render_metric_card("CPU Hardware", f"{cpu_cores} Cores", f"{proc_str}", icon="🖥️")
    with c4:
        render_metric_card("Selected Workers", f"[{workers_text}]", "Active allocation list", icon="⚙️")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card("Fastest Time", fastest_time_str, fastest_mode_str, icon="⚡")
    with c6:
        render_metric_card("Peak Speedup", best_speedup_str, best_speedup_sub, icon="🚀")
    with c7:
        render_metric_card("Optimization Gain", gain_label, gain_sub, icon="🏆")
    with c8:
        if monitor_summary:
            avg_cpu = monitor_summary.get("avg_cpu_percent", 0.0)
            peak_ram = monitor_summary.get("peak_memory_mb", 0.0)
            render_metric_card("Resource Usage", f"{avg_cpu:.1f}% CPU", f"Peak RAM: {peak_ram:.0f} MB", icon="📊")
        else:
            repeats_val = int(df["runs"].iloc[0]) if df is not None and not df.empty else params.get("repeats", 3)
            render_metric_card("Benchmark Repeats", f"{repeats_val} Runs", "Per configuration trial", icon="🔁")

    st.markdown("---")

    # 3. Dynamic Status & Guidance
    if df is not None and not df.empty:
        st.markdown(
            f"""
            <div class="hpc-callout-success">
                <b>Benchmark Active:</b> Completed execution across <b>{len(df)} configurations</b>.
                Explore the <b>Performance Analytics</b> tab for 3-way comparative scaling curves,
                <b>Scalability & Crossover</b> to detect parallel acceleration thresholds,
                or <b>Reports</b> to download the publication-quality PDF report.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if intelligence and intelligence.get("key_takeaways"):
            with st.expander("💡 Performance Intelligence Summary & Key Findings", expanded=True):
                st.markdown(f"**Recommended Configuration:** {intelligence.get('recommendation_reason', '')}")
                st.markdown("**Key Takeaways:**")
                for item in intelligence.get("key_takeaways", []):
                    st.markdown(f"- {item}", unsafe_allow_html=True)
    elif loaded_images:
        st.markdown(
            """
            <div class="hpc-callout-info">
                <b>Images Loaded & Ready:</b> Configure filter parameters and worker counts in the sidebar,
                then click <b>▶️ Run Benchmark</b> to evaluate Sequential, Original Parallel, and Optimized Parallel pipelines.
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
            2. **Original Parallel Multiprocessing**: Standard Python `multiprocessing.Pool` with `spawn` start method,
               where input and output NumPy arrays are serialized (pickled) across OS pipes.
            3. **Optimized Parallel Multiprocessing**: High-performance pipeline leveraging POSIX `multiprocessing.shared_memory`
               for zero-copy array transfers, reusable persistent worker pools, single-threaded BLAS to prevent CPU oversubscription,
               and adaptive chunking.
            4. **Speedup ($S_p$) & Efficiency ($E_p$)**:
               $$\\text{Speedup } S_p = \\frac{T_1}{T_p}, \\quad \\text{Efficiency } E_p = \\frac{S_p}{p}$$
            5. **Parallel Crossover Threshold**: The workload size where multi-core computational throughput overtakes
               the fixed latency of process management and inter-process communication.
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


def render_analytics_tab(
    df: Optional[pd.DataFrame],
    last_results: Optional[List[Dict[str, Any]]] = None,
    intelligence: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Performance Analytics tab with 3-way comparative charts, throughput metrics, and worker PID distribution."""
    st.markdown("### ⚡ HPC Scaling & Performance Analytics")

    if df is None or df.empty:
        st.info("👈 Run the benchmark from the sidebar to generate execution time curves, speedup charts, and efficiency plots.")
        return

    has_opt = "Optimized Parallel" in df["mode"].values

    # Highlights separating fastest overall from best parallel configuration
    st.markdown("#### 🏆 Performance Highlights")
    fastest_overall_idx = df["mean_time_s"].idxmin()
    fastest_overall_row = df.loc[fastest_overall_idx]

    par_modes = ["Parallel", "Optimized Parallel", "Original Parallel"]
    par_df = df[df["mode"].isin(par_modes)]

    h_col1, h_col2, h_col3 = st.columns(3)
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
                    <div class="hpc-card-title">🚀 Peak Measured Speedup</div>
                    <div class="hpc-card-value">{float(best_sp_row['speedup']):.2f}x Speedup</div>
                    <div class="hpc-card-subvalue"><b>Mode:</b> {best_sp_row['mode']} ({int(best_sp_row['workers'])}w) | <b>Eff:</b> {float(best_sp_row['efficiency']) * 100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="hpc-card">
                    <div class="hpc-card-title">🚀 Peak Measured Speedup</div>
                    <div class="hpc-card-value">1.00x</div>
                    <div class="hpc-card-subvalue">Sequential only</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with h_col3:
        if has_opt:
            opt_df = df[df["mode"] == "Optimized Parallel"]
            max_gain = float(opt_df["optimization_gain"].max()) if not opt_df.empty else 1.0
            st.markdown(
                f"""
                <div class="hpc-card" style="border-left: 4px solid #8b5cf6;">
                    <div class="hpc-card-title">⚡ Peak Optimization Gain</div>
                    <div class="hpc-card-value">{max_gain:.2f}x Faster</div>
                    <div class="hpc-card-subvalue">SharedMemory vs. Original Parallel</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="hpc-card">
                    <div class="hpc-card-title">⚡ Benchmark Protocol</div>
                    <div class="hpc-card-value">Standard Mode</div>
                    <div class="hpc-card-subvalue">Run 3-Way Benchmark for comparisons</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Performance Intelligence Recommendation Box
    if intelligence:
        st.markdown(
            f"""
            <div class="hpc-callout-info">
                <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.35rem;">
                    💡 Performance Intelligence Recommendation
                </div>
                <div>{intelligence.get('recommendation_reason', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Performance Charts
    st.markdown("#### 📈 Scaling Visualizations")
    if has_opt:
        tab_time, tab_speedup, tab_eff, tab_tp = st.tabs(
            [
                "3-Way Execution Time",
                "Speedup Curves",
                "Parallel Efficiency",
                "Pipeline Throughput",
            ]
        )

        with tab_time:
            fig_time = plot_three_way_execution_time(df)
            st.pyplot(fig_time)
            plt.close(fig_time)
            st.caption(
                "**3-Way Execution Time Comparison:** Contrasts single-core Sequential baseline against Original Multiprocessing "
                "(array pickling over pipes) and Optimized Multiprocessing (POSIX SharedMemory zero-copy)."
            )

        with tab_speedup:
            fig_speedup = plot_speedup_comparison(df)
            st.pyplot(fig_speedup)
            plt.close(fig_speedup)
            st.caption(
                "**Parallel Speedup Curves:** Compares empirical speedup against ideal theoretical linear scaling (S = p). "
                "Points above 1.0 indicate wall-clock acceleration over sequential execution."
            )

        with tab_eff:
            fig_eff = plot_efficiency_comparison(df)
            st.pyplot(fig_eff)
            plt.close(fig_eff)
            st.caption(
                "**Parallel Core Efficiency:** Measures compute utilization percentage per worker core ($E_p = S_p / p$). "
                "Dashed 100% line represents ideal scaling."
            )

        with tab_tp:
            fig_tp = plot_throughput_comparison(df)
            st.pyplot(fig_tp)
            plt.close(fig_tp)
            st.caption(
                "**Pipeline Throughput:** Demonstrates processing capacity in Images per second and Megapixels per second."
            )

    else:
        tab_time, tab_speedup, tab_eff = st.tabs(["Execution Time", "Speedup Curve", "Parallel Efficiency"])

        with tab_time:
            fig_time = plot_execution_time(df)
            st.pyplot(fig_time)
            plt.close(fig_time)

        with tab_speedup:
            fig_speedup = plot_speedup(df)
            st.pyplot(fig_speedup)
            plt.close(fig_speedup)

        with tab_eff:
            fig_eff = plot_efficiency(df)
            st.pyplot(fig_eff)
            plt.close(fig_eff)

    st.markdown("---")

    # Complete Metrics Table
    st.markdown("#### 📋 Complete Benchmark Metrics Table")
    display_df = df.copy()

    # Format numbers for clean presentation
    display_df["mean_time_s"] = display_df["mean_time_s"].map(lambda x: f"{float(x):.3f}s")
    display_df["min_time_s"] = display_df["min_time_s"].map(lambda x: f"{float(x):.3f}s")
    display_df["std_time_s"] = display_df["std_time_s"].map(lambda x: f"{float(x):.3f}s")
    display_df["speedup"] = display_df["speedup"].map(lambda x: f"{float(x):.2f}x")
    display_df["efficiency"] = display_df["efficiency"].map(lambda x: f"{float(x) * 100:.1f}%")

    if "throughput_images_s" in display_df.columns:
        display_df["throughput_images_s"] = display_df["throughput_images_s"].map(lambda x: f"{float(x):.1f}")
    if "throughput_mp_s" in display_df.columns:
        display_df["throughput_mp_s"] = display_df["throughput_mp_s"].map(lambda x: f"{float(x):.2f}")
    if "optimization_gain" in display_df.columns:
        display_df["optimization_gain"] = display_df["optimization_gain"].map(lambda x: f"{float(x):.2f}x")

    st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    # Worker Process Task Allocation (Real PIDs)
    if last_results:
        st.markdown("#### 👥 Worker Process Task Allocation")
        st.write(
            "Visual inspection of actual worker OS Process IDs (PIDs) that executed image processing tasks, "
            "confirming data-parallel load distribution."
        )
        col_w_chart, col_w_table = st.columns([1, 1])
        with col_w_chart:
            fig_worker = plot_worker_task_distribution(last_results)
            st.pyplot(fig_worker)
            plt.close(fig_worker)

        with col_w_table:
            w_df = aggregate_worker_distribution(last_results)
            if not w_df.empty:
                w_disp = w_df.copy()
                w_disp.columns = [
                    "Worker PID",
                    "Tasks Processed",
                    "Total Time (s)",
                    "Mean Time / Img (s)",
                    "Task Share (%)",
                ]
                w_disp["Total Time (s)"] = w_disp["Total Time (s)"].map(lambda x: f"{x:.3f}")
                w_disp["Mean Time / Img (s)"] = w_disp["Mean Time / Img (s)"].map(lambda x: f"{x:.3f}")
                w_disp["Task Share (%)"] = w_disp["Task Share (%)"].map(lambda x: f"{x:.1f}%")
                st.dataframe(w_disp, use_container_width=True)


def render_scalability_tab(
    params: Optional[Dict[str, Any]] = None,
    default_workers: Optional[List[int]] = None,
    system_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Workload Scalability & Parallel Crossover Analysis tab."""
    if params is None:
        params = {"kernel_size": 5, "sigma": 1.0, "repeats": 1}
    st.markdown("### 📈 Workload Scalability & Parallel Crossover Analysis")
    st.write(
        "Evaluate how parallel speedup scales with increasing workload size (image counts and resolutions), "
        "and empirically identify the **parallel crossover threshold** where multi-core execution overtakes sequential processing."
    )

    # 1. Configuration Panel
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        res_choice = st.selectbox(
            "Test Resolution:",
            options=["512 × 512 px (Fast)", "1024 × 1024 px (Standard)", "2048 × 2048 px (HD Intensive)"],
            index=1,
            key="scalability_res_choice",
        )
        if "512" in res_choice:
            width, height = 512, 512
        elif "1024" in res_choice:
            width, height = 1024, 1024
        else:
            width, height = 2048, 2048

        workers_to_test = st.selectbox(
            "Parallel Worker Process Count:",
            options=[w for w in [2, 4, 6, 8, 10] if w <= (os.cpu_count() or 1)],
            index=1,
            key="scalability_workers_choice",
        )

    with col_sc2:
        image_counts_choice = st.multiselect(
            "Workload Image Counts to Sweep:",
            options=[4, 8, 12, 16, 24, 32, 48, 64],
            default=[4, 8, 16, 24],
            key="scalability_counts_choice",
        )
        sc_mode = st.radio(
            "Measurement Protocol:",
            options=["Steady-State (Pre-warmed Workers)", "End-to-End (Process Creation Included)"],
            index=0,
            horizontal=True,
            key="scalability_mode_choice",
        )
        sc_benchmark_mode = "steady_state" if "Steady-State" in sc_mode else "end_to_end"

    # 2. Automated Memory Safety Estimation
    if image_counts_choice:
        max_c = max(image_counts_choice)
        mem_est = estimate_memory_requirements(max_c, width, height, workers_to_test)

        st.markdown(
            f"""
            <div style="margin: 0.5rem 0 1rem 0;">
                <span class="hpc-meta-pill">💾 <b>Peak RAM Required:</b> ~{mem_est['estimated_peak_mb']:.0f} MB</span>
                <span class="hpc-meta-pill">🟢 <b>Available RAM:</b> {mem_est['available_mb']:.0f} MB</span>
                <span class="hpc-meta-pill">🛡️ <b>Safety Status:</b> {'✅ Safe Headroom' if mem_est['is_safe'] else '⚠️ High Memory Warning'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if mem_est["warning_message"]:
            st.warning(mem_est["warning_message"])

    # 3. Execution Trigger
    run_sc_btn = st.button("▶️ Run Scalability Sweep & Detect Crossover", type="primary", use_container_width=True)

    if run_sc_btn:
        if not image_counts_choice:
            st.warning("Please select at least one image count to sweep.")
        else:
            sorted_counts = sorted(image_counts_choice)
            workload_configs = [
                {
                    "label": f"{cnt} imgs ({width}x{height})",
                    "count": cnt,
                    "width": width,
                    "height": height,
                }
                for cnt in sorted_counts
            ]

            p_bar = st.progress(0, text="Starting scalability experiment sweep...")
            status_txt = st.empty()

            def on_sc_progress(done: int, total: int, msg: str) -> None:
                p_bar.progress(done / total)
                status_txt.caption(f"⚡ Step {done}/{total}: {msg}")

            with st.spinner("Running scalability sweep across workload configurations..."):
                try:
                    sc_df, crossover_info = run_scalability_experiment(
                        workload_configs=workload_configs,
                        params=params,
                        num_workers=workers_to_test,
                        benchmark_mode=sc_benchmark_mode,
                        progress_callback=on_sc_progress,
                    )
                    st.session_state["scalability_df"] = sc_df
                    st.session_state["crossover_info"] = crossover_info
                    status_txt.empty()
                    st.success("✅ Scalability sweep completed successfully!")
                except Exception as exc:
                    st.error(f"❌ Scalability experiment failed: {exc}")

    # 4. Results Display
    saved_sc_df = st.session_state.get("scalability_df")
    saved_crossover = st.session_state.get("crossover_info")

    if saved_sc_df is not None and not saved_sc_df.empty:
        st.markdown("---")
        st.markdown("#### 📊 Scalability Results & Crossover Findings")

        if saved_crossover:
            if saved_crossover.get("crossover_observed"):
                st.markdown(
                    f"""
                    <div class="hpc-callout-success">
                        <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;">
                            🎯 Parallel Performance Crossover Observed!
                        </div>
                        <div>{saved_crossover.get('analysis_text', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="hpc-callout-info">
                        <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem;">
                            ℹ️ Parallel Crossover Analysis
                        </div>
                        <div>{saved_crossover.get('analysis_text', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Plot Crossover Curve
        fig_cross = plot_scalability_crossover(saved_sc_df)
        st.pyplot(fig_cross)
        plt.close(fig_cross)

        # Scalability Data Table
        st.markdown("#### 📋 Scalability Sweep Metrics Table")
        st.dataframe(saved_sc_df, use_container_width=True)


def render_system_monitor_tab(
    system_info: Dict[str, Any],
    monitor_history: Optional[pd.DataFrame] = None,
    monitor_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the System Monitor tab with hardware specs, real-time gauges, and telemetry timelines."""
    st.markdown("### 🖥️ System Monitor & Hardware Profiling")
    st.write(
        "Real-time CPU and memory telemetry recorded during benchmark execution, combined with "
        "cross-platform hardware auto-detection."
    )

    # 1. Hardware Specifications Panel
    st.markdown("#### 💻 Hardware Specifications")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        proc_name = system_info.get("processor", "Apple Silicon")
        render_metric_card("Processor", str(proc_name), system_info.get("platform", "macOS"), icon="🧠")
    with m2:
        arch = system_info.get("machine", "arm64")
        phys = system_info.get("physical_cores", system_info.get("cpu_count", "N/A"))
        log = system_info.get("logical_cores", system_info.get("cpu_count", "N/A"))
        render_metric_card("CPU Cores", f"{phys}P / {log}L", f"Architecture: {arch}", icon="🏗️")
    with m3:
        total_ram = system_info.get("total_ram_gb")
        avail_ram = system_info.get("available_ram_gb")
        ram_label = f"{total_ram:.1f} GB Total" if total_ram else "N/A"
        ram_sub = f"{avail_ram:.1f} GB Available" if avail_ram else "RAM"
        render_metric_card("System Memory", ram_label, ram_sub, icon="💾")
    with m4:
        py_ver = system_info.get("python_version", "3.12")
        sm = system_info.get("start_method", "spawn")
        render_metric_card("Runtime", f"Python {py_ver}", f"MP Start: {sm}", icon="🐍")

    st.markdown("---")

    # 2. Real-Time Instantaneous Gauges
    st.markdown("#### ⚡ Live System Resource Telemetry")
    live_c1, live_c2 = st.columns(2)

    current_cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    current_ram_pct = vm.percent
    current_ram_used_gb = vm.used / (1024.0**3)

    with live_c1:
        st.markdown(f"**Current System CPU Utilization:** `{current_cpu:.1f}%`")
        st.progress(min(1.0, current_cpu / 100.0))

    with live_c2:
        st.markdown(f"**Current System RAM Utilization:** `{current_ram_pct:.1f}%` ({current_ram_used_gb:.2f} GB used)")
        st.progress(min(1.0, current_ram_pct / 100.0))

    st.markdown("---")

    # 3. Post-Benchmark Monitoring Telemetry Timelines
    st.markdown("#### 📊 Monitored Benchmark Resource Telemetry")
    if monitor_history is not None and not monitor_history.empty and len(monitor_history) > 1:
        if monitor_summary:
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                render_metric_card("Monitored Duration", f"{monitor_summary.get('duration_seconds', 0.0):.2f} s", f"{monitor_summary.get('samples_count', 0)} samples collected", icon="⏱️")
            with t2:
                render_metric_card("Average CPU", f"{monitor_summary.get('avg_cpu_percent', 0.0):.1f}%", f"Peak: {monitor_summary.get('peak_cpu_percent', 0.0):.1f}%", icon="⚡")
            with t3:
                render_metric_card("Peak Memory", f"{monitor_summary.get('peak_memory_mb', 0.0):.0f} MB", f"{monitor_summary.get('peak_memory_percent', 0.0):.1f}% of total RAM", icon="📈")
            with t4:
                render_metric_card("Memory Growth", f"{monitor_summary.get('memory_growth_mb', 0.0):.1f} MB", "During pool execution", icon="🌱")

            if monitor_summary.get("high_memory_warning"):
                st.warning("⚠️ **High Memory Utilization Warning:** RAM usage exceeded 85% during benchmark execution.")

        tab_cpu_plot, tab_mem_plot = st.tabs(["CPU Utilization Timeline", "RAM Consumption Timeline"])
        with tab_cpu_plot:
            fig_cpu = plot_cpu_utilization(monitor_history)
            st.pyplot(fig_cpu)
            plt.close(fig_cpu)

        with tab_mem_plot:
            fig_mem = plot_memory_utilization(monitor_history)
            st.pyplot(fig_mem)
            plt.close(fig_mem)
    else:
        st.info("👈 Run a benchmark from the sidebar with **Enable Real-Time Monitoring** checked to view telemetry charts.")

    st.markdown("---")

    # 4. Architecture & Resource Dynamics
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
                    operating at lower clock frequencies, introducing scheduling desynchronization where fast cores wait for slower cores.
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
                    Under the <code>spawn</code> method, each worker is an independent Python interpreter process.
                    Passing raw image buffers requires serializing (pickling) NumPy arrays across OS pipes,
                    consuming CPU memory bandwidth. <b>Module 11 SharedMemory</b> eliminates this serialization bottleneck completely.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_tab(
    df: Optional[pd.DataFrame],
    current_exp_id: Optional[int] = None,
) -> None:
    """Render the Benchmark History tab with SQLite experiment archive, filtering, and export."""
    st.markdown("### 📜 Benchmark History & SQLite Experiment Archive")
    st.write(
        "Persistent local database (`results/benchmark_history.db`) tracking all historical benchmark experiments, "
        "workload parameters, scaling metrics, and hardware telemetry."
    )

    history_df = get_all_experiments()

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        render_metric_card("Total Experiments", f"{len(history_df)}", "Logged in SQLite", icon="🗄️")
    with h2:
        cum_mp = history_df["total_megapixels"].sum() if not history_df.empty else 0.0
        render_metric_card("Total Workload", f"{cum_mp:.1f} MP", "Cumulative processed", icon="📐")
    with h3:
        best_sp = history_df["max_speedup"].max() if not history_df.empty else 1.0
        render_metric_card("Peak Recorded Speedup", f"{best_sp:.2f}x", "Across all runs", icon="🚀")
    with h4:
        best_eff = history_df["best_efficiency"].max() * 100.0 if not history_df.empty else 100.0
        render_metric_card("Peak Recorded Efficiency", f"{best_eff:.1f}%", "Optimal scaling", icon="🏆")

    st.markdown("---")

    if history_df.empty:
        st.info("No benchmark experiments logged yet. Run a benchmark to record your first experiment!")
        return

    # Filter Controls
    st.markdown("#### 🔍 Filter & Search Experiments")
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        max_workload = float(history_df["total_megapixels"].max()) if not history_df.empty else 1.0
        min_mp = st.slider("Minimum Workload (Megapixels):", min_value=0.0, max_value=max(1.0, max_workload), value=0.0, step=0.1)
    with f_c2:
        mode_filter = st.selectbox("Fastest Mode:", options=["All", "Optimized Parallel", "Parallel", "Sequential"])

    filtered_df = history_df[history_df["total_megapixels"] >= min_mp]
    if mode_filter != "All":
        filtered_df = filtered_df[filtered_df["fastest_mode"] == mode_filter]

    st.markdown(f"**Showing {len(filtered_df)} of {len(history_df)} experiments:**")
    st.dataframe(filtered_df, use_container_width=True)

    # Multi-Run Trend Chart
    if len(filtered_df) >= 2:
        st.markdown("#### 📈 Multi-Run Scaling Trends")
        fig_trend, ax_trend = plt.subplots(figsize=(8, 3.5), dpi=150)
        ax_trend.plot(
            filtered_df["id"],
            filtered_df["max_speedup"],
            marker="o",
            color="#2563eb",
            linewidth=2.0,
            label="Peak Speedup (x)",
        )
        ax_trend.axhline(1.0, color="#dc2626", linestyle="--", alpha=0.7, label="Sequential Baseline (1.0x)")
        ax_trend.set_xlabel("Experiment ID", fontsize=10)
        ax_trend.set_ylabel("Speedup Factor", fontsize=10)
        ax_trend.set_title("Peak Speedup Progression Across Experiments", fontsize=11, fontweight="bold")
        ax_trend.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
        ax_trend.legend(loc="upper left", fontsize=9)
        fig_trend.tight_layout()
        st.pyplot(fig_trend)
        plt.close(fig_trend)

    st.markdown("---")

    # Inspect Specific Experiment
    st.markdown("#### 🔬 Inspect Experiment Details")
    selected_id = st.selectbox("Select Experiment ID to inspect:", options=history_df["id"].tolist())
    if selected_id:
        exp_details = get_experiment_details(int(selected_id))
        if exp_details:
            st.write(f"**Experiment #{selected_id}** — Recorded: `{exp_details.get('timestamp')}`")
            m_df = exp_details.get("metrics_df")
            if m_df is not None and not m_df.empty:
                st.dataframe(m_df, use_container_width=True)
            mon = exp_details.get("monitor_summary")
            if mon:
                st.caption(f"Hardware Telemetry: Avg CPU: {mon.get('avg_cpu_percent')}% | Peak RAM: {mon.get('peak_memory_mb')} MB")

    st.markdown("---")

    # History Actions (CSV Export, Delete, Clear)
    st.markdown("#### ⚙️ History Archive Actions")
    act_c1, act_c2, act_c3 = st.columns(3)

    with act_c1:
        csv_bytes = export_history_to_csv_bytes()
        st.download_button(
            "📥 Export Full History (CSV)",
            data=csv_bytes,
            file_name="benchmark_experiments_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with act_c2:
        if st.button("🗑️ Delete Selected Experiment", use_container_width=True):
            if delete_experiment(int(selected_id)):
                st.success(f"Deleted experiment #{selected_id}")
                st.rerun()

    with act_c3:
        confirm_clear = st.checkbox("Confirm Clear All History", key="chk_confirm_clear_hist")
        if confirm_clear:
            if st.button("⚠️ Clear Entire History DB", use_container_width=True, type="primary"):
                clear_all_experiments()
                st.success("All historical experiments cleared.")
                st.rerun()


def render_reports_tab(
    df: Optional[pd.DataFrame],
    last_results: Optional[List[Dict[str, Any]]],
    loaded_images: List[Tuple[str, np.ndarray]],
    params: Dict[str, Any],
    system_info: Dict[str, Any],
    monitor_summary: Optional[Dict[str, Any]] = None,
    monitor_history: Optional[pd.DataFrame] = None,
    intelligence: Optional[Dict[str, Any]] = None,
    scalability_df: Optional[pd.DataFrame] = None,
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
        "hardware specifications, experiment configuration, 3-way comparative tabular metrics, throughput tables, "
        "worker PID distributions, scaling graphs, real-time telemetry timelines, workload crossover analysis, "
        "automated performance intelligence, and the full 4-stage processed image gallery for all uploaded images."
    )

    pdf_bytes = generate_benchmark_pdf(
        df=df,
        last_results=last_results,
        original_images=loaded_images,
        params=params,
        system_info=system_info,
        monitor_summary=monitor_summary,
        monitor_history=monitor_history,
        intelligence=intelligence,
        scalability_df=scalability_df,
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

    has_opt = "Optimized Parallel" in df["mode"].values
    if has_opt:
        f_time = plot_three_way_execution_time(df)
        f_sp = plot_speedup_comparison(df)
        f_eff = plot_efficiency_comparison(df)
    else:
        f_time = plot_execution_time(df)
        f_sp = plot_speedup(df)
        f_eff = plot_efficiency(df)

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
