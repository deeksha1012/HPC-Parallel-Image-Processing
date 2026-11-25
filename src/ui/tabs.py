"""
Modular Streamlit tab renderers for the HPC Image Processing Lab.

Organizes the application interface into 6 dedicated, clean views:
1. Overview: Executive summary cards, workload guide, and performance intelligence highlights.
2. Image Processing: 4-stage gallery, interactive before/after comparison slider, and full-resolution inspection.
3. Performance Analytics: Scaling charts, speedup curves, throughput metrics, worker PID distribution, and intelligence recommendations.
4. System Monitor: Cross-platform hardware profiling, real-time CPU/RAM gauges, post-benchmark telemetry charts, and architectural insights.
5. Benchmark History: SQLite persistent experiment archive, filtering, multi-run trend charts, experiment inspection, and CSV export.
6. Reports: Export center for comprehensive PDF report, CSV data, ZIP image archive, and Matplotlib graphs.
"""

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
        render_metric_card("Best Configuration", best_config_str, best_config_sub, icon="🏆")
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
                Explore the <b>Image Processing</b> tab for filter comparisons, <b>Performance Analytics</b> for speedup curves & worker distribution,
                <b>System Monitor</b> for live/telemetry profiles, or <b>Reports</b> to download the publication-quality PDF report.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Performance Intelligence Callout Banner on Overview
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
            4. **Throughput Metrics**:
               $$\\text{Throughput}_{\\text{images}} = \\frac{N}{T_p} \\quad (\\text{images/s}), \\quad \\text{Throughput}_{\\text{MP}} = \\frac{\\text{MP}}{T_p} \\quad (\\text{MP/s})$$
            5. **Amdahl's Law & IPC Friction**: For small image sets, inter-process pickling and process creation
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


def render_analytics_tab(
    df: Optional[pd.DataFrame],
    last_results: Optional[List[Dict[str, Any]]] = None,
    intelligence: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Performance Analytics tab with scaling charts, throughput metrics, worker PID distribution, and intelligence."""
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

    # HPC Throughput & Variability Table
    st.markdown("#### 🚀 HPC Throughput & Runtime Variability")
    st.write(
        "Throughput quantifies work completed per unit time. Coefficient of Variation ($CV = \\sigma / \\mu \\times 100\\%$) "
        "measures runtime stability across repeated trials."
    )
    df_aug = compute_throughput_metrics(df)
    tp_display = df_aug[
        [
            "mode",
            "workers",
            "mean_time_s",
            "throughput_images_per_s",
            "throughput_mp_per_s",
            "cv_percent",
            "ideal_speedup",
            "speedup_deficit",
        ]
    ].copy()
    tp_display.columns = [
        "Mode",
        "Workers",
        "Mean Time (s)",
        "Throughput (Images/s)",
        "Throughput (MP/s)",
        "Variability (CV %)",
        "Ideal Speedup",
        "Speedup Deficit",
    ]
    tp_display["Mean Time (s)"] = tp_display["Mean Time (s)"].map(lambda x: f"{x:.3f}")
    tp_display["Throughput (Images/s)"] = tp_display["Throughput (Images/s)"].map(lambda x: f"{x:.2f}")
    tp_display["Throughput (MP/s)"] = tp_display["Throughput (MP/s)"].map(lambda x: f"{x:.2f}")
    tp_display["Variability (CV %)"] = tp_display["Variability (CV %)"].map(lambda x: f"{x:.2f}%")
    tp_display["Ideal Speedup"] = tp_display["Ideal Speedup"].map(lambda x: f"{x:.1f}x")
    tp_display["Speedup Deficit"] = tp_display["Speedup Deficit"].map(lambda x: f"{x:.2f}x")
    st.dataframe(tp_display, use_container_width=True)

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
        # Telemetry summary cards
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
            st.caption(
                "Shows system-wide CPU utilization alongside individual execution cores over the duration of the benchmark."
            )

        with tab_mem_plot:
            fig_mem = plot_memory_utilization(monitor_history)
            st.pyplot(fig_mem)
            plt.close(fig_mem)
            st.caption(
                "Tracks resident set memory footprint and RAM percentage over time."
            )
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
                    consuming CPU memory bandwidth before filtering computation begins.
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

    # History Summary Cards
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
        mode_filter = st.selectbox("Fastest Mode:", options=["All", "Parallel", "Sequential"])

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
        "hardware specifications, experiment configuration, complete tabular metrics, throughput tables, "
        "worker PID distributions, scaling graphs, real-time telemetry timelines, automated performance intelligence, "
        "and the full 4-stage processed image gallery for all uploaded images."
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
