"""
SQLite Benchmark Experiment Archive and History Management.

Provides persistent storage, querying, filtering, and export for benchmark
experiments, parameters, timing metrics, and hardware resource summaries.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import RESULTS_DIR

DEFAULT_DB_PATH = RESULTS_DIR / "benchmark_history.db"


def get_db_path(custom_path: Optional[Path | str] = None) -> Path:
    """Resolve and ensure parent directory for the SQLite database path."""
    if custom_path is not None:
        p = Path(custom_path)
    else:
        p = DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db(db_path: Optional[Path | str] = None) -> Path:
    """
    Initialize SQLite database and create the experiments table if absent.

    Args:
        db_path: Optional custom path to SQLite database file.

    Returns:
        Path: Path to initialized SQLite database file.
    """
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                num_images INTEGER NOT NULL,
                total_megapixels REAL NOT NULL,
                kernel_size INTEGER NOT NULL,
                sigma REAL NOT NULL,
                repeats INTEGER NOT NULL,
                worker_counts TEXT NOT NULL,
                fastest_mode TEXT NOT NULL,
                fastest_workers INTEGER NOT NULL,
                fastest_time_s REAL NOT NULL,
                seq_time_s REAL NOT NULL,
                max_speedup REAL NOT NULL,
                best_efficiency REAL NOT NULL,
                avg_cpu_percent REAL,
                peak_memory_mb REAL,
                notes TEXT,
                metrics_json TEXT NOT NULL,
                monitor_summary_json TEXT
            )
            """
        )
        conn.commit()
    return path


def save_experiment(
    df: pd.DataFrame,
    params: Dict[str, Any],
    num_images: int,
    total_megapixels: float,
    monitor_summary: Optional[Dict[str, Any]] = None,
    notes: str = "",
    db_path: Optional[Path | str] = None,
) -> int:
    """
    Persist a benchmark experiment and its execution metrics into SQLite.

    Args:
        df: Benchmark DataFrame with results across modes and worker counts.
        params: Filter settings dict ('kernel_size', 'sigma', 'repeats').
        num_images: Count of images evaluated.
        total_megapixels: Total workload size in MP.
        monitor_summary: Optional background resource monitor summary.
        notes: Optional user note or experiment description.
        db_path: Optional custom SQLite database path.

    Returns:
        int: The primary key ID of the inserted experiment record.
    """
    if df.empty:
        raise ValueError("Cannot save an empty benchmark DataFrame.")

    path = init_db(db_path)

    # 1. Compute summary attributes
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    kernel_size = int(params.get("kernel_size", 15))
    sigma = float(params.get("sigma", 3.0))
    repeats = int(df["runs"].iloc[0]) if "runs" in df.columns else int(params.get("repeats", 3))

    seq_rows = df[df["mode"] == "Sequential"]
    seq_time_s = float(seq_rows["mean_time_s"].iloc[0]) if not seq_rows.empty else 0.0

    fastest_idx = df["mean_time_s"].idxmin()
    fastest_row = df.loc[fastest_idx]
    fastest_mode = str(fastest_row["mode"])
    fastest_workers = int(fastest_row["workers"])
    fastest_time_s = float(fastest_row["mean_time_s"])

    par_df = df[df["mode"] == "Parallel"]
    if not par_df.empty:
        max_speedup = float(par_df["speedup"].max())
        best_efficiency = float(par_df["efficiency"].max())
        workers_list = sorted(par_df["workers"].astype(int).unique().tolist())
        worker_counts_str = ", ".join(map(str, workers_list))
    else:
        max_speedup = 1.0
        best_efficiency = 1.0
        worker_counts_str = "None (Sequential Only)"

    avg_cpu_percent = (
        float(monitor_summary.get("avg_cpu_percent", 0.0)) if monitor_summary else None
    )
    peak_memory_mb = (
        float(monitor_summary.get("peak_memory_mb", 0.0)) if monitor_summary else None
    )

    metrics_json = json.dumps(df.to_dict(orient="records"))
    monitor_summary_json = json.dumps(monitor_summary) if monitor_summary else None

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (
                timestamp, num_images, total_megapixels, kernel_size, sigma, repeats,
                worker_counts, fastest_mode, fastest_workers, fastest_time_s, seq_time_s,
                max_speedup, best_efficiency, avg_cpu_percent, peak_memory_mb, notes,
                metrics_json, monitor_summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                num_images,
                round(total_megapixels, 4),
                kernel_size,
                sigma,
                repeats,
                worker_counts_str,
                fastest_mode,
                fastest_workers,
                round(fastest_time_s, 4),
                round(seq_time_s, 4),
                round(max_speedup, 2),
                round(best_efficiency, 4),
                round(avg_cpu_percent, 2) if avg_cpu_percent is not None else None,
                round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
                notes,
                metrics_json,
                monitor_summary_json,
            ),
        )
        conn.commit()
        exp_id = cursor.lastrowid

    return exp_id or 0


def get_all_experiments(db_path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Retrieve all benchmark experiment records ordered by timestamp descending.

    Returns:
        pd.DataFrame: Table of experiment runs, or empty DataFrame if none exist.
    """
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, timestamp, num_images, total_megapixels, kernel_size, sigma, repeats,
                   worker_counts, fastest_mode, fastest_workers, fastest_time_s, seq_time_s,
                   max_speedup, best_efficiency, avg_cpu_percent, peak_memory_mb, notes
            FROM experiments
            ORDER BY id DESC
            """,
            conn,
        )
    return df


def get_experiment_details(
    exp_id: int,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete details of a single experiment including deserialized JSON.

    Args:
        exp_id: Primary key ID of the experiment.
        db_path: Optional custom SQLite database path.

    Returns:
        Optional[Dict[str, Any]]: Record dictionary with 'metrics_df' and 'monitor_summary'.
    """
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()

    if not row:
        return None

    record = dict(row)
    # Parse JSON fields
    try:
        raw_metrics = json.loads(record.get("metrics_json", "[]"))
        record["metrics_df"] = pd.DataFrame(raw_metrics)
    except Exception:
        record["metrics_df"] = pd.DataFrame()

    try:
        raw_mon = record.get("monitor_summary_json")
        record["monitor_summary"] = json.loads(raw_mon) if raw_mon else None
    except Exception:
        record["monitor_summary"] = None

    return record


def delete_experiment(exp_id: int, db_path: Optional[Path | str] = None) -> bool:
    """
    Delete a specific experiment by ID.

    Returns:
        bool: True if deleted successfully, False otherwise.
    """
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_experiments(db_path: Optional[Path | str] = None) -> bool:
    """
    Clear all experiment records from the database.

    Returns:
        bool: True if successful.
    """
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments")
        conn.commit()
        return True


def export_history_to_csv_bytes(db_path: Optional[Path | str] = None) -> bytes:
    """
    Export all experiment records to in-memory UTF-8 CSV bytes.

    Returns:
        bytes: CSV encoded byte string ready for download.
    """
    df = get_all_experiments(db_path)
    return df.to_csv(index=False).encode("utf-8")
