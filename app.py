"""
HPC Image Processing Lab — Parallel Computing and Performance Analysis Platform.

Main Streamlit application implementing Module 10: HPC System Monitoring and Performance Intelligence.
Provides a modern laboratory interface organized into 6 functional tabs:
1. Overview: Executive summary cards, workload metrics, performance intelligence, and workflow guide.
2. Image Processing: 4-stage gallery, interactive before/after comparison slider, and full-resolution inspection.
3. Performance Analytics: Scaling charts, speedup curves, throughput metrics, and worker PID allocation.
4. System Monitor: Cross-platform hardware profiling, real-time gauges, and resource telemetry charts.
5. Benchmark History: SQLite experiment archive, filtering, multi-run trends, and CSV export.
6. Reports: Comprehensive PDF report, CSV metrics table, ZIP archives, and graph exports.
"""

import os
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from src.benchmark import get_system_info, run_benchmark
from src.config import (
    DEFAULT_KERNEL_SIZE,
    DEFAULT_REPEATS,
    DEFAULT_SIGMA,
    MAX_UPLOAD_IMAGES,
    SUPPORTED_EXTENSIONS,
)
from src.history import save_experiment
from src.image_io import load_image
from src.intelligence import analyze_performance
from src.monitor import ResourceMonitor, get_detailed_hardware_info
from src.sample_generator import (
    RESOLUTION_PRESETS,
    generate_sample_images_in_memory,
)
from src.ui.components import inject_custom_css
from src.ui.tabs import (
    render_analytics_tab,
    render_history_tab,
    render_image_processing_tab,
    render_overview_tab,
    render_reports_tab,
    render_system_monitor_tab,
)


def load_uploaded_files(uploaded_files: list) -> List[Tuple[str, np.ndarray]]:
    """
    Load uploaded files with Pillow into RGB uint8 NumPy arrays.

    Handles file count limits, skips unreadable files with a warning,
    and deduplicates duplicate file names.
    """
    if len(uploaded_files) > MAX_UPLOAD_IMAGES:
        st.sidebar.warning(
            f"Uploaded {len(uploaded_files)} images. Only the first "
            f"{MAX_UPLOAD_IMAGES} will be processed."
        )

    seen_names: dict[str, int] = {}
    loaded: List[Tuple[str, np.ndarray]] = []

    for file in uploaded_files[:MAX_UPLOAD_IMAGES]:
        name = file.name
        if name in seen_names:
            seen_names[name] += 1
            stem = Path(name).stem
            ext = Path(name).suffix
            name = f"{stem}_{seen_names[name]}{ext}"
        else:
            seen_names[name] = 1

        try:
            arr = load_image(file)
            loaded.append((name, arr))
        except Exception as exc:
            st.warning(f"Skipping '{file.name}': {exc}")

    return loaded


def main() -> None:
    """Main application routine building the HPC Image Processing Lab."""
    # 1. Page Configuration
    st.set_page_config(
        page_title="HPC Image Processing Lab",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Inject modern laboratory CSS styling
    inject_custom_css()

    # Initialize session state stores if absent
    if "loaded_images" not in st.session_state:
        st.session_state["loaded_images"] = []
    if "system_info" not in st.session_state:
        st.session_state["system_info"] = get_system_info()

    # 3. Sidebar: Application Controls and Configuration
    st.sidebar.markdown("## 🔬 HPC Image Processing Lab")
    st.sidebar.markdown(
        '<span class="hpc-badge">Module 10: System Telemetry & Intelligence</span>',
        unsafe_allow_html=True,
    )

    # Reset Experiment Action
    if st.sidebar.button(
        "🔄 Reset Experiment",
        use_container_width=True,
        help="Clear all uploaded images, generated samples, and benchmark data",
    ):
        for key in [
            "loaded_images",
            "benchmark_df",
            "last_results",
            "original_images",
            "params",
            "monitor_summary",
            "monitor_history",
            "intelligence",
            "current_exp_id",
        ]:
            st.session_state.pop(key, None)
        st.session_state["loaded_images"] = []
        st.rerun()

    st.sidebar.markdown("---")

    # Image Input Modes (Sample Generator vs. Custom Upload)
    st.sidebar.markdown("### 📂 Image Dataset")
    input_mode = st.sidebar.radio(
        "Choose image source:",
        options=["🎨 Generate Sample Images", "📁 Upload Custom Images"],
        index=0 if not st.session_state.get("uploaded_files_present", False) else 1,
        help="Use synthetic reproducible test patterns or upload your own PNG/JPG images.",
    )

    if input_mode == "🎨 Generate Sample Images":
        st.sidebar.caption("Generate synthetic images with gradients, shapes, and noise:")
        preset_choice = st.sidebar.selectbox(
            "Resolution preset:",
            options=list(RESOLUTION_PRESETS.keys()),
            index=1,
            help="Select resolution preset balancing processing speed and workload size.",
        )
        sample_count = st.sidebar.slider(
            "Image count:",
            min_value=1,
            max_value=16,
            value=4,
            help="Number of synthetic sample images to generate.",
        )
        sample_seed = st.sidebar.number_input(
            "Random seed (reproducibility):",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="Base seed for deterministic gradient and shape generation.",
        )

        if st.sidebar.button("✨ Generate Synthetic Images", use_container_width=True):
            width, height = RESOLUTION_PRESETS[preset_choice]
            with st.spinner("Generating synthetic test images in memory..."):
                gen_images = generate_sample_images_in_memory(
                    count=sample_count,
                    width=width,
                    height=height,
                    seed=int(sample_seed),
                )
                st.session_state["loaded_images"] = gen_images
                st.session_state.pop("benchmark_df", None)
                st.session_state.pop("last_results", None)
                st.session_state.pop("monitor_summary", None)
                st.session_state.pop("monitor_history", None)
                st.session_state.pop("intelligence", None)
                st.sidebar.success(f"Generated {len(gen_images)} images ({width}×{height})!")
                st.rerun()

    else:
        ext_list = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
        uploaded = st.sidebar.file_uploader(
            "Upload image files:",
            type=ext_list,
            accept_multiple_files=True,
            help="Supported file formats: PNG, JPG, JPEG (converted to RGB uint8).",
        )
        if uploaded:
            st.session_state["uploaded_files_present"] = True
            st.session_state["loaded_images"] = load_uploaded_files(uploaded)

    # Dataset Summary Indicator
    loaded_images = st.session_state.get("loaded_images", [])
    if loaded_images:
        total_mp = sum(a.shape[0] * a.shape[1] for _, a in loaded_images) / 1_000_000.0
        st.sidebar.info(f"Active dataset: **{len(loaded_images)} image(s)** ({total_mp:.2f} Megapixels)")

    st.sidebar.markdown("---")

    # Filter & Multiprocessing Parameters with Tooltips
    st.sidebar.markdown("### ⚙️ Pipeline Parameters")

    kernel_size = st.sidebar.select_slider(
        "Gaussian Kernel Size:",
        options=[i for i in range(3, 33, 2)],
        value=DEFAULT_KERNEL_SIZE,
        help="Size of the square Gaussian blur filter window in pixels. Must be an odd number (3 to 31) so that the filter has an exact center pixel.",
    )
    sigma = st.sidebar.slider(
        "Gaussian Sigma (σ):",
        min_value=0.5,
        max_value=10.0,
        value=float(DEFAULT_SIGMA),
        step=0.5,
        help="Standard deviation for the Gaussian distribution. Controls the spatial spread of the blur filter.",
    )

    cpu_max = os.cpu_count() or 1
    worker_options = list(range(1, cpu_max + 1))
    default_workers = sorted(list(set([w for w in [2, 4, cpu_max] if w in worker_options])))
    worker_counts = st.sidebar.multiselect(
        "Worker process counts to benchmark:",
        options=worker_options,
        default=default_workers,
        help="List of parallel CPU worker counts to evaluate. Each count tests data-parallel execution using Python multiprocessing (spawn start method).",
    )

    repeats = st.sidebar.slider(
        "Benchmark repeats per test:",
        min_value=1,
        max_value=5,
        value=int(DEFAULT_REPEATS),
        step=1,
        help="Number of times each configuration is repeated to calculate a stable mean and standard deviation.",
    )

    # HPC Telemetry Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ HPC Monitoring")
    enable_monitoring = st.sidebar.checkbox(
        "Enable Real-Time Monitoring",
        value=True,
        help="Sample background CPU and RAM utilization during benchmark runs using psutil.",
    )
    sampling_interval = st.sidebar.slider(
        "Sampling Interval (s):",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Polling interval for recording CPU and memory consumption during benchmarking.",
    )

    # Benchmark Execution Trigger
    st.sidebar.markdown("---")
    run_btn = st.sidebar.button(
        "▶️ Run Benchmark",
        type="primary",
        use_container_width=True,
        help="Execute sequential baseline and all selected parallel configurations.",
    )

    # 4. Main Area Header & Laboratory Banner
    st.markdown(
        """
        <div class="hpc-main-header">
            <div class="hpc-title">HPC Image Processing Lab</div>
            <div class="hpc-subtitle">Parallel Computing and Performance Analysis Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Handle Benchmark Execution
    if run_btn:
        if not loaded_images:
            st.warning("⚠️ Please generate sample images or upload image files in the sidebar before running.")
        elif not worker_counts:
            st.warning("⚠️ Please select at least one worker process count in the sidebar.")
        else:
            params = {
                "kernel_size": kernel_size,
                "sigma": sigma,
                "repeats": repeats,
            }

            # Interactive live progress bar with configuration & step details
            progress_bar = st.progress(0, text="Initializing benchmark runs...")
            status_text = st.empty()

            def on_progress(done: int, total: int, msg: str) -> None:
                progress_bar.progress(done / total)
                status_text.caption(f"⚡ **Task Progress:** Step {done} of {total} — {msg}")

            monitor = ResourceMonitor(interval=sampling_interval) if enable_monitoring else None
            if monitor is not None:
                monitor.start()

            with st.spinner("Running sequential baseline and parallel worker benchmarks..."):
                try:
                    df, last_results = run_benchmark(
                        images=loaded_images,
                        params=params,
                        worker_counts=worker_counts,
                        repeats=repeats,
                        progress_callback=on_progress,
                    )
                except Exception as exc:
                    st.error(f"❌ Benchmark execution failed: {exc}")
                    df, last_results = None, None
                finally:
                    if monitor is not None:
                        monitor.stop()

            if df is not None and last_results is not None:
                monitor_summary = monitor.get_summary() if monitor else None
                monitor_history = monitor.get_history_df() if monitor else pd.DataFrame()
                sys_info = get_system_info()
                intel = analyze_performance(df, monitor_summary, sys_info)

                # Persist experiment to SQLite
                total_pixels = sum(arr.shape[0] * arr.shape[1] for _, arr in loaded_images)
                total_mp = round(total_pixels / 1_000_000.0, 4)
                exp_id = save_experiment(
                    df=df,
                    params=params,
                    num_images=len(loaded_images),
                    total_megapixels=total_mp,
                    monitor_summary=monitor_summary,
                )

                st.session_state["benchmark_df"] = df
                st.session_state["last_results"] = last_results
                st.session_state["original_images"] = loaded_images
                st.session_state["params"] = params
                st.session_state["system_info"] = sys_info
                st.session_state["monitor_summary"] = monitor_summary
                st.session_state["monitor_history"] = monitor_history
                st.session_state["intelligence"] = intel
                st.session_state["current_exp_id"] = exp_id
                status_text.empty()
                st.success("✅ Benchmark execution completed successfully across all configurations!")

    # 6. Retrieve active session state data
    df = st.session_state.get("benchmark_df", None)
    last_results = st.session_state.get("last_results", None)
    current_images = st.session_state.get("original_images", loaded_images)
    params = st.session_state.get(
        "params",
        {"kernel_size": kernel_size, "sigma": sigma, "repeats": repeats},
    )
    system_info = st.session_state.get("system_info", get_system_info())
    monitor_summary = st.session_state.get("monitor_summary", None)
    monitor_history = st.session_state.get("monitor_history", None)
    intelligence = st.session_state.get("intelligence", None)
    current_exp_id = st.session_state.get("current_exp_id", None)

    # 7. Navigation Tabs
    tab_overview, tab_processing, tab_analytics, tab_monitor, tab_history, tab_reports = st.tabs(
        [
            "📊 Overview",
            "🖼️ Image Processing",
            "⚡ Performance Analytics",
            "🖥️ System Monitor",
            "📜 Benchmark History",
            "📑 Reports",
        ]
    )

    with tab_overview:
        render_overview_tab(
            df=df,
            last_results=last_results,
            loaded_images=current_images,
            system_info=system_info,
            params=params,
            selected_workers=worker_counts,
            intelligence=intelligence,
            monitor_summary=monitor_summary,
        )

    with tab_processing:
        render_image_processing_tab(
            loaded_images=current_images,
            last_results=last_results,
            params=params,
        )

    with tab_analytics:
        render_analytics_tab(
            df=df,
            last_results=last_results,
            intelligence=intelligence,
        )

    with tab_monitor:
        render_system_monitor_tab(
            system_info=system_info,
            monitor_history=monitor_history,
            monitor_summary=monitor_summary,
        )

    with tab_history:
        render_history_tab(
            df=df,
            current_exp_id=current_exp_id,
        )

    with tab_reports:
        render_reports_tab(
            df=df,
            last_results=last_results,
            loaded_images=current_images,
            params=params,
            system_info=system_info,
            monitor_summary=monitor_summary,
            monitor_history=monitor_history,
            intelligence=intelligence,
        )


if __name__ == "__main__":
    main()
