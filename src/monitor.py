"""
HPC System Monitoring and Hardware Telemetry Module.

Provides:
1. Detailed cross-platform hardware auto-detection (Apple Silicon aware).
2. Real-time background CPU and RAM telemetry sampling during benchmark runs.
3. Worker process PID task distribution aggregation and plotting.
4. Matplotlib figure generators for CPU and memory utilization timelines.
"""

from datetime import datetime
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil


def get_processor_name() -> str:
    """
    Detect the processor brand name across platforms without hardcoding.

    On macOS (Apple Silicon / Intel), queries sysctl 'machdep.cpu.brand_string'.
    On Linux, reads /proc/cpuinfo model name.
    On Windows, reads the PROCESSOR_IDENTIFIER environment variable.
    Falls back gracefully to platform.processor() or platform.machine().
    """
    sys_name = platform.system()
    if sys_name == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sys_name == "Windows":
        proc_id = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if proc_id:
            return proc_id

    # Fallback to standard platform queries
    proc = platform.processor()
    if proc:
        return proc
    return f"{platform.machine()} Processor"


def get_detailed_hardware_info() -> Dict[str, Any]:
    """
    Collect comprehensive hardware and operating system specifications.

    Uses platform, os, and psutil to detect processor model, architecture,
    physical/logical core counts, total and available RAM, and Python runtime.

    Returns:
        Dict[str, Any]: Hardware metadata dictionary.
    """
    vm = psutil.virtual_memory()
    total_ram_gb = round(vm.total / (1024.0**3), 2)
    avail_ram_gb = round(vm.available / (1024.0**3), 2)
    used_ram_gb = round(vm.used / (1024.0**3), 2)
    ram_percent = vm.percent

    # Physical vs Logical cores
    logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores is None or physical_cores <= 0:
        physical_cores = logical_cores

    processor_name = get_processor_name()
    machine_arch = platform.machine()
    os_name = f"{platform.system()} {platform.release()}"

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "processor": processor_name,
        "machine": machine_arch,
        "platform": os_name,
        "full_platform": platform.platform(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "cpu_count": logical_cores,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": avail_ram_gb,
        "used_ram_gb": used_ram_gb,
        "ram_percent": ram_percent,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "start_method": "spawn",
    }


class ResourceMonitor:
    """
    Lightweight, thread-safe background resource monitor.

    Samples overall CPU utilization, per-core CPU utilization, and virtual
    memory metrics at a user-configured interval during benchmark execution.
    """

    def __init__(self, interval: float = 0.2):
        """
        Initialize the resource monitor.

        Args:
            interval: Sampling interval in seconds (clamped to [0.05, 1.0]).
        """
        self.interval = max(0.05, min(1.0, float(interval)))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def start(self) -> None:
        """Start the background sampling thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._records.clear()
            self._stop_event.clear()
            self._start_time = time.perf_counter()
            self._end_time = 0.0

            # Prime psutil.cpu_percent with non-blocking initial call
            psutil.cpu_percent(interval=None)
            psutil.cpu_percent(percpu=True, interval=None)

            self._thread = threading.Thread(
                target=self._sample_loop,
                name="HPC-ResourceMonitor",
                daemon=True,
            )
            self._thread.start()

    def _sample_loop(self) -> None:
        """Internal worker loop running in the background thread."""
        while not self._stop_event.is_set():
            sample_time = time.perf_counter() - self._start_time
            try:
                cpu_total = psutil.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
                vm = psutil.virtual_memory()

                record = {
                    "time_s": round(sample_time, 3),
                    "cpu_percent": float(cpu_total),
                    "memory_percent": float(vm.percent),
                    "memory_used_mb": round(vm.used / (1024.0 * 1024.0), 2),
                    "memory_avail_mb": round(vm.available / (1024.0 * 1024.0), 2),
                }

                # Store per-core percentages with core_0, core_1, ...
                for i, c_pct in enumerate(cpu_per_core):
                    record[f"core_{i}"] = float(c_pct)

                with self._lock:
                    self._records.append(record)
            except Exception:
                pass

            # Wait for next interval or early exit
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        """Stop sampling and record completion timestamp."""
        with self._lock:
            if not self._stop_event.is_set():
                self._stop_event.set()
                self._end_time = time.perf_counter()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_running(self) -> bool:
        """Return True if the background sampler thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def get_history_df(self) -> pd.DataFrame:
        """
        Return the time-series measurements as a pandas DataFrame.

        Returns:
            pd.DataFrame: Columns include time_s, cpu_percent, memory_percent,
                          memory_used_mb, memory_avail_mb, and core_0..core_N.
        """
        with self._lock:
            if not self._records:
                # Return empty DataFrame with standard columns
                return pd.DataFrame(
                    columns=[
                        "time_s",
                        "cpu_percent",
                        "memory_percent",
                        "memory_used_mb",
                        "memory_avail_mb",
                    ]
                )
            return pd.DataFrame(self._records)

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute aggregate resource utilization metrics over the monitored duration.

        Returns:
            Dict[str, Any]: Summary dictionary with average/peak CPU %,
                            per-core averages, baseline and peak memory, and warnings.
        """
        df = self.get_history_df()
        if df.empty:
            # Fallback instantaneous reading
            vm = psutil.virtual_memory()
            return {
                "duration_seconds": 0.0,
                "samples_count": 0,
                "avg_cpu_percent": 0.0,
                "peak_cpu_percent": 0.0,
                "min_cpu_percent": 0.0,
                "avg_per_cpu_percent": [],
                "baseline_memory_mb": round(vm.used / (1024.0 * 1024.0), 2),
                "peak_memory_mb": round(vm.used / (1024.0 * 1024.0), 2),
                "memory_growth_mb": 0.0,
                "peak_memory_percent": float(vm.percent),
                "high_memory_warning": vm.percent > 85.0,
            }

        duration = (
            (self._end_time - self._start_time)
            if self._end_time > self._start_time
            else float(df["time_s"].iloc[-1])
        )
        avg_cpu = float(df["cpu_percent"].mean())
        peak_cpu = float(df["cpu_percent"].max())
        min_cpu = float(df["cpu_percent"].min())

        # Per core averages
        core_cols = [c for c in df.columns if c.startswith("core_")]
        avg_per_core = [float(df[col].mean()) for col in core_cols]

        baseline_mem = float(df["memory_used_mb"].iloc[0])
        peak_mem = float(df["memory_used_mb"].max())
        mem_growth = max(0.0, peak_mem - baseline_mem)
        peak_mem_pct = float(df["memory_percent"].max())

        return {
            "duration_seconds": round(duration, 3),
            "samples_count": len(df),
            "avg_cpu_percent": round(avg_cpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "min_cpu_percent": round(min_cpu, 2),
            "avg_per_cpu_percent": [round(x, 2) for x in avg_per_core],
            "baseline_memory_mb": round(baseline_mem, 2),
            "peak_memory_mb": round(peak_mem, 2),
            "memory_growth_mb": round(mem_growth, 2),
            "peak_memory_percent": round(peak_mem_pct, 2),
            "high_memory_warning": peak_mem_pct > 85.0,
        }


def aggregate_worker_distribution(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate execution metrics by worker Process ID (PID).

    Calculates the count of tasks, total computation time, and mean duration
    per worker process using real PID data collected during image processing.

    Args:
        results: List of result dicts containing 'pid', 'name', and 'time_seconds'.

    Returns:
        pd.DataFrame: Columns ['pid', 'tasks_processed', 'total_time_s', 'mean_time_s', 'task_share_pct']
    """
    if not results:
        return pd.DataFrame(
            columns=["pid", "tasks_processed", "total_time_s", "mean_time_s", "task_share_pct"]
        )

    pid_data: Dict[int, List[float]] = {}
    for r in results:
        pid = int(r.get("pid", os.getpid()))
        t = float(r.get("time_seconds", 0.0))
        if pid not in pid_data:
            pid_data[pid] = []
        pid_data[pid].append(t)

    total_tasks = len(results)
    rows = []
    for pid, times in sorted(pid_data.items(), key=lambda item: len(item[1]), reverse=True):
        count = len(times)
        total_t = sum(times)
        mean_t = total_t / count if count > 0 else 0.0
        share_pct = (count / total_tasks * 100.0) if total_tasks > 0 else 0.0
        rows.append(
            {
                "pid": pid,
                "tasks_processed": count,
                "total_time_s": round(total_t, 4),
                "mean_time_s": round(mean_t, 4),
                "task_share_pct": round(share_pct, 1),
            }
        )

    return pd.DataFrame(rows)


def plot_cpu_utilization(
    history_df: pd.DataFrame,
    title: str = "System & Core CPU Utilization Over Time",
) -> plt.Figure:
    """
    Generate a publication-grade line chart displaying CPU utilization over time.

    Plots the overall system CPU percentage along with transparent traces for
    individual CPU execution cores.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)

    if history_df.empty or "cpu_percent" not in history_df.columns:
        ax.text(
            0.5,
            0.5,
            "No CPU telemetry recorded for this run",
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig

    time_series = history_df["time_s"]
    core_cols = [c for c in history_df.columns if c.startswith("core_")]

    # Plot individual per-core traces with subtle opacity
    for col in core_cols:
        core_num = col.split("_")[1]
        ax.plot(
            time_series,
            history_df[col],
            linewidth=0.9,
            alpha=0.35,
            label=f"Core {core_num}",
        )

    # Plot bold system total line
    ax.plot(
        time_series,
        history_df["cpu_percent"],
        linewidth=2.4,
        color="#2563eb",
        label="Overall System CPU",
        zorder=5,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax.set_ylabel("CPU Utilization (%)", fontsize=10)
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

    # Limit legend items if there are many cores
    if len(core_cols) <= 8:
        ax.legend(loc="upper right", fontsize=8, ncol=min(4, len(core_cols) + 1))
    else:
        # Show only system CPU in legend to prevent clutter
        ax.legend([ax.get_lines()[-1]], ["Overall System CPU"], loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def plot_memory_utilization(
    history_df: pd.DataFrame,
    title: str = "RAM Utilization Over Time",
) -> plt.Figure:
    """
    Generate a line chart illustrating virtual memory (RAM) usage and percentage over time.
    """
    fig, ax1 = plt.subplots(figsize=(8, 3.8), dpi=150)

    if history_df.empty or "memory_used_mb" not in history_df.columns:
        ax1.text(
            0.5,
            0.5,
            "No Memory telemetry recorded for this run",
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
            transform=ax1.transAxes,
        )
        ax1.set_title(title, fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig

    time_series = history_df["time_s"]
    color_mb = "#059669"  # Emerald green
    color_pct = "#d97706"  # Amber

    # Axis 1: Memory Used in Megabytes
    ax1.plot(
        time_series,
        history_df["memory_used_mb"],
        color=color_mb,
        linewidth=2.2,
        label="Used RAM (MB)",
    )
    ax1.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax1.set_ylabel("RAM Used (MB)", color=color_mb, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color_mb)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

    # Axis 2: RAM Percentage
    ax2 = ax1.twinx()
    ax2.plot(
        time_series,
        history_df["memory_percent"],
        color=color_pct,
        linewidth=1.8,
        linestyle="--",
        label="RAM Utilization (%)",
    )
    ax2.set_ylabel("RAM Utilization (%)", color=color_pct, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_pct)
    ax2.set_ylim(0, 100)

    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_worker_task_distribution(results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Generate a bar chart illustrating task distribution across worker process PIDs.
    """
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)

    agg_df = aggregate_worker_distribution(results)
    if agg_df.empty:
        ax.text(
            0.5,
            0.5,
            "No worker process results available",
            ha="center",
            va="center",
            fontsize=11,
            color="#64748b",
            transform=ax.transAxes,
        )
        ax.set_title("Worker Process Task Allocation", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig

    x_labels = [f"PID {p}" for p in agg_df["pid"]]
    counts = agg_df["tasks_processed"]
    bars = ax.bar(x_labels, counts, color="#3b82f6", edgecolor="#1d4ed8", width=0.55)

    # Label counts on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)} img",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Worker Process Task Allocation (Images Processed)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Worker Process ID (PID)", fontsize=10)
    ax.set_ylabel("Images Processed", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.25 + 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")

    fig.tight_layout()
    return fig
