"""
Performance metrics for parallel computing.

Defines standard High Performance Computing (HPC) metric calculations:
1. Speedup (S): How much faster parallel execution is compared to sequential.
2. Parallel Efficiency (E): How effectively additional CPU cores are utilized.
"""


def calculate_speedup(t_seq: float, t_par: float) -> float:
    """
    Calculate the parallel speedup factor.

    Formula:
        Speedup S = T_seq / T_par

    Where:
        T_seq = Execution time of the sequential baseline algorithm.
        T_par = Execution time of the parallel algorithm using p workers.

    Args:
        t_seq (float): Sequential execution time in seconds.
        t_par (float): Parallel execution time in seconds.

    Returns:
        float: Speedup factor S. Values > 1.0 indicate parallel acceleration.

    Raises:
        ValueError: If either t_seq or t_par is less than or equal to zero.
    """
    if t_seq <= 0.0:
        raise ValueError(f"Sequential time must be greater than zero, got {t_seq}.")
    if t_par <= 0.0:
        raise ValueError(f"Parallel time must be greater than zero, got {t_par}.")

    return float(t_seq / t_par)


def calculate_efficiency(speedup: float, workers: int) -> float:
    """
    Calculate the parallel efficiency.

    Formula:
        Efficiency E = S / p

    Where:
        S = Speedup factor (T_seq / T_par).
        p = Number of worker processes (CPU cores assigned).

    An efficiency of 1.0 (or 100%) indicates linear (ideal) speedup.
    Real-world efficiency is typically below 1.0 due to multiprocessing overhead,
    IPC communication latency, and memory bandwidth contention.

    Args:
        speedup (float): Calculated speedup factor.
        workers (int): Number of worker processes (p >= 1).

    Returns:
        float: Parallel efficiency E as a fraction (1.0 = 100% ideal).

    Raises:
        ValueError: If workers is less than 1.
    """
    if workers < 1:
        raise ValueError(f"Worker count must be at least 1, got {workers}.")

    return float(speedup / workers)
