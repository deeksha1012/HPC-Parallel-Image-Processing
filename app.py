"""
Streamlit Web Dashboard for High Performance Image Processing.

Provides an interactive user interface to upload images, configure Gaussian blur
and core allocation parameters, run sequential and parallel benchmarks, inspect
execution times and efficiency metrics, visualize image processing stages,
display performance charts, and export results in various formats.
"""

import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
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
from src.export import (
    create_results_zip,
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
    image_to_png_bytes,
)
from src.image_io import load_image
from src.plots import plot_efficiency, plot_execution_time, plot_speedup


def load_uploaded_files(uploaded_files: list) -> list[tuple[str, np.ndarray]]:
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
    loaded: list[tuple[str, np.ndarray]] = []

    for file in uploaded_files[:MAX_UPLOAD_IMAGES]:
        name = file.name
        # Make duplicate filenames unique so result lookups stay consistent
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


def render_summary_metrics(df: pd.DataFrame) -> None:
    """Display top-level performance cards for sequential time, parallel speedup, and efficiency."""
    seq_row = df[df["mode"] == "Sequential"].iloc[0]
    par_df = df[df["mode"] == "Parallel"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sequential Time", f"{seq_row['mean_time_s']:.3f} s")

    if not par_df.empty:
        fastest_idx = par_df["mean_time_s"].idxmin()
        fastest_row = par_df.loc[fastest_idx]
        max_sp_row = par_df.loc[par_df["speedup"].idxmax()]

        col2.metric(
            "Fastest Parallel",
            f"{fastest_row['mean_time_s']:.3f} s",
            f"{int(fastest_row['workers'])} workers",
        )
        col3.metric("Highest Speedup", f"{max_sp_row['speedup']:.2f}x")
        col4.metric(
            "Parallel Efficiency",
            f"{fastest_row['efficiency'] * 100:.1f}%",
        )

        if fastest_row["mean_time_s"] >= seq_row["mean_time_s"]:
            st.info(
                "Notice: Sequential processing was faster than parallel in this run. "
                "For small image batches, process spawning and memory transfer overhead "
                "can outweigh parallel computation gains."
            )


def render_results_table(df: pd.DataFrame) -> None:
    """Display the benchmark metrics table with clear headers and formatted numbers."""
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

    st.subheader("Benchmark Metrics Table")
    st.dataframe(display_df, use_container_width=True)


def render_processed_images(
    last_results: list[dict[str, Any]],
    original_images: list[tuple[str, np.ndarray]],
) -> str:
    """
    Display a 4-column visual gallery comparing Original, Grayscale, Blur, and Edges.

    Returns:
        str: Currently selected image name for targeted downloads.
    """
    st.subheader("Processed Images Preview")
    orig_dict = {name: arr for name, arr in original_images}
    res_dict = {res["name"]: res for res in last_results}

    image_names = list(orig_dict.keys())
    if not image_names:
        return ""

    selected = st.selectbox("Select an image to inspect", options=image_names)
    if selected and selected in res_dict:
        c1, c2, c3, c4 = st.columns(4)
        c1.image(orig_dict[selected], caption="1. Original RGB", use_container_width=True)
        c2.image(res_dict[selected]["gray"], caption="2. Grayscale", use_container_width=True)
        c3.image(res_dict[selected]["blur"], caption="3. Gaussian Blur", use_container_width=True)
        c4.image(res_dict[selected]["edges"], caption="4. Sobel Edges", use_container_width=True)

    return selected or ""


def render_performance_graphs(df: pd.DataFrame) -> None:
    """Render performance graphs in tabs with captions and close figures cleanly."""
    st.subheader("Performance Graphs")
    tab_time, tab_speedup, tab_eff = st.tabs(["Execution time", "Speedup", "Efficiency"])

    with tab_time:
        fig_time = plot_execution_time(df)
        st.pyplot(fig_time)
        plt.close(fig_time)
        st.caption(
            "How to read: Lower bars represent faster total execution time. Error bars show "
            "the standard deviation across runs, illustrating execution stability."
        )

    with tab_speedup:
        fig_speedup = plot_speedup(df)
        st.pyplot(fig_speedup)
        plt.close(fig_speedup)
        st.caption(
            "How to read: Values above 1.0 indicate acceleration over sequential execution. "
            "The dashed line represents theoretical ideal linear speedup where performance "
            "scales proportionally with worker core count."
        )

    with tab_eff:
        fig_eff = plot_efficiency(df)
        st.pyplot(fig_eff)
        plt.close(fig_eff)
        st.caption(
            "How to read: Measures how effectively additional worker cores contribute to speedup. "
            "The dashed line at 100% shows ideal linear scaling; real efficiency decreases with "
            "more cores due to inter-process communication and synchronization overhead."
        )


def render_downloads(
    df: pd.DataFrame,
    last_results: list[dict[str, Any]],
    selected_image_name: str,
) -> None:
    """Render export and download options for images, archives, tables, and graphs."""
    st.subheader("Downloads")

    # 1. Download outputs for currently selected image
    if selected_image_name:
        res_dict = {res["name"]: res for res in last_results}
        if selected_image_name in res_dict:
            res = res_dict[selected_image_name]
            stem = Path(selected_image_name).stem
            st.markdown(f"**Download individual outputs for:** `{selected_image_name}`")
            col_g, col_b, col_e = st.columns(3)

            col_g.download_button(
                "Download Grayscale (PNG)",
                data=image_to_png_bytes(res["gray"]),
                file_name=f"{stem}_gray.png",
                mime="image/png",
                key="dl_single_gray",
            )
            col_b.download_button(
                "Download Gaussian Blur (PNG)",
                data=image_to_png_bytes(res["blur"]),
                file_name=f"{stem}_blur.png",
                mime="image/png",
                key="dl_single_blur",
            )
            col_e.download_button(
                "Download Sobel Edges (PNG)",
                data=image_to_png_bytes(res["edges"]),
                file_name=f"{stem}_edges.png",
                mime="image/png",
                key="dl_single_edges",
            )

    st.divider()

    # 2. Batch ZIP download with stage selection
    st.markdown("**Download all processed images (ZIP Archive)**")
    zip_stages = st.multiselect(
        "Select output types to include in ZIP archive:",
        options=["gray", "blur", "edges"],
        default=["gray", "blur", "edges"],
        key="zip_stages_multiselect",
    )
    if zip_stages:
        zip_bytes = create_results_zip(last_results, which=zip_stages)
        st.download_button(
            "Download all processed images (ZIP)",
            data=zip_bytes,
            file_name="processed_images.zip",
            mime="application/zip",
            key="dl_all_images_zip",
        )

    st.divider()

    # 3. Download results table as CSV
    st.markdown("**Download benchmark data**")
    csv_bytes = dataframe_to_csv_bytes(df)
    st.download_button(
        "Download results table (CSV)",
        data=csv_bytes,
        file_name="benchmark_results.csv",
        mime="text/csv",
        key="dl_benchmark_csv",
    )

    st.divider()

    # 4. Download high-resolution graph PNGs
    st.markdown("**Download performance graphs (PNG)**")
    c_g1, c_g2, c_g3 = st.columns(3)

    f_time = plot_execution_time(df)
    b_time = figure_to_png_bytes(f_time)
    plt.close(f_time)
    c_g1.download_button(
        "Download Execution Time Graph",
        data=b_time,
        file_name="execution_time.png",
        mime="image/png",
        key="dl_fig_time",
    )

    f_sp = plot_speedup(df)
    b_sp = figure_to_png_bytes(f_sp)
    plt.close(f_sp)
    c_g2.download_button(
        "Download Speedup Graph",
        data=b_sp,
        file_name="speedup.png",
        mime="image/png",
        key="dl_fig_speedup",
    )

    f_eff = plot_efficiency(df)
    b_eff = figure_to_png_bytes(f_eff)
    plt.close(f_eff)
    c_g3.download_button(
        "Download Efficiency Graph",
        data=b_eff,
        file_name="efficiency.png",
        mime="image/png",
        key="dl_fig_efficiency",
    )


def render_system_info(info: dict[str, Any]) -> None:
    """Display hardware and software specifications in an expandable section."""
    with st.expander("Test machine details"):
        st.write(f"- **Timestamp:** {info.get('timestamp')}")
        st.write(f"- **Platform:** {info.get('platform')}")
        st.write(f"- **Machine Architecture:** {info.get('machine')}")
        st.write(f"- **Python Version:** {info.get('python_version')}")
        st.write(f"- **Total CPU Cores:** {info.get('cpu_count')}")
        st.write(f"- **Start Method:** {info.get('start_method')}")


def main() -> None:
    """Main application routine building the Streamlit dashboard."""
    st.set_page_config(page_title="HPC Image Processing", layout="wide")

    st.title("High Performance Image Processing Using Parallel Computing")
    st.write(
        "This project applies grayscale conversion, Gaussian blur, and Sobel edge detection "
        "to batches of images sequentially and in parallel using Python multiprocessing. "
        "It benchmarks execution times, speedup factors, and parallel efficiency across core allocations."
    )
    st.caption(
        "Note: Parallel is not always faster: with few or small images, process start-up "
        "and data copying can outweigh the gain."
    )

    # Sidebar: Parameter controls and file uploader
    st.sidebar.header("Configuration")
    ext_list = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.sidebar.file_uploader(
        "Upload images",
        type=ext_list,
        accept_multiple_files=True,
    )

    kernel_size = st.sidebar.select_slider(
        "Gaussian Kernel Size",
        options=[i for i in range(3, 33, 2)],
        value=DEFAULT_KERNEL_SIZE,
    )
    sigma = st.sidebar.slider(
        "Gaussian Sigma",
        min_value=0.5,
        max_value=10.0,
        value=float(DEFAULT_SIGMA),
        step=0.5,
    )

    cpu_max = os.cpu_count() or 1
    worker_options = list(range(1, cpu_max + 1))
    default_workers = sorted(list(set([w for w in [2, 4, cpu_max] if w in worker_options])))
    worker_counts = st.sidebar.multiselect(
        "Worker process counts to benchmark",
        options=worker_options,
        default=default_workers,
    )

    repeats = st.sidebar.slider(
        "Benchmark repeats per test",
        min_value=1,
        max_value=5,
        value=int(DEFAULT_REPEATS),
        step=1,
    )
    run_btn = st.sidebar.button("Run benchmark", type="primary")

    # Load and validate uploaded images
    loaded_images = load_uploaded_files(uploaded_files) if uploaded_files else []
    if loaded_images:
        total_mp = sum(a.shape[0] * a.shape[1] for _, a in loaded_images) / 1_000_000.0
        st.sidebar.info(f"Loaded {len(loaded_images)} image(s) ({total_mp:.2f} Megapixels).")

    # Handle Run button trigger
    if run_btn:
        if not loaded_images:
            st.warning("Please upload at least one supported image in the sidebar before running.")
        elif not worker_counts:
            st.warning("Please select at least one worker process count in the sidebar.")
        else:
            params = {"kernel_size": kernel_size, "sigma": sigma}
            progress_bar = st.progress(0, text="Starting benchmark runs...")

            def on_progress(done: int, total: int, msg: str) -> None:
                progress_bar.progress(done / total, text=msg)

            with st.spinner("Running benchmarks across sequential and parallel configurations..."):
                try:
                    df, last_results = run_benchmark(
                        images=loaded_images,
                        params=params,
                        worker_counts=worker_counts,
                        repeats=repeats,
                        progress_callback=on_progress,
                    )
                    # Persist benchmark outputs in session state across UI reruns
                    st.session_state["benchmark_df"] = df
                    st.session_state["last_results"] = last_results
                    st.session_state["original_images"] = loaded_images
                    st.session_state["system_info"] = get_system_info()
                    st.success("Benchmark completed successfully!")
                except Exception as exc:
                    st.error(f"Benchmark error: {exc}")

    # Display results from session state if available
    if "benchmark_df" in st.session_state:
        st.divider()
        df = st.session_state["benchmark_df"]
        last_results = st.session_state["last_results"]
        orig_images = st.session_state["original_images"]

        render_summary_metrics(df)
        render_results_table(df)
        selected_img = render_processed_images(last_results, orig_images)
        render_performance_graphs(df)
        render_downloads(df, last_results, selected_img)
        render_system_info(st.session_state["system_info"])
    else:
        st.info(
            "Upload one or more images in the sidebar, configure settings, "
            "and click **Run benchmark** to view real performance metrics."
        )


if __name__ == "__main__":
    main()
