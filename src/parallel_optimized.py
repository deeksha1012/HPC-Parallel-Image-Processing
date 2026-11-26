"""
Optimized Data-Parallel Batch Image Processing Engine for HPC Workloads.

Implements high-performance multiprocessing optimizations:
1. Zero-copy Inter-Process Communication via multiprocessing.shared_memory (POSIX shared memory).
2. Elimination of multi-megabyte NumPy array pickling across IPC pipes.
3. Reusable, persistent worker pools to amortize spawn initialization latency.
4. Prevention of NumPy / Apple Accelerate internal thread oversubscription.
5. Adaptive task batching and chunk sizing.
6. Robust shared memory allocation and deterministic teardown without resource leaks.
"""

from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.filters import gaussian_blur, sobel_edges, to_grayscale


def _init_worker() -> None:
    """
    Worker process initializer.

    Restricts BLAS, OpenMP, and Apple Accelerate frameworks to single-threaded
    execution inside each worker process to prevent CPU core oversubscription
    and thread thrashing across physical performance cores.
    """
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _worker_shm_task(task_args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Top-level picklable task executed inside worker processes.

    Reads the input RGB image directly from shared memory with a zero-copy
    NumPy buffer view, executes the transformation pipeline, and writes output
    filter stages directly into shared memory output segments.

    Args:
        task_args: Tuple containing:
            - task_id (int)
            - name (str)
            - height (int)
            - width (int)
            - in_shm_name (str)
            - in_offset (int)
            - out_gray_name (str)
            - out_blur_name (str)
            - out_edges_name (str)
            - out_offset (int)
            - kernel_size (int)
            - sigma (float)

    Returns:
        Dict[str, Any]: Lightweight metadata payload (no heavy image arrays).
    """
    (
        task_id,
        name,
        h,
        w,
        in_shm_name,
        in_offset,
        out_gray_name,
        out_blur_name,
        out_edges_name,
        out_offset,
        kernel_size,
        sigma,
    ) = task_args

    t_start = time.perf_counter()

    # 1. Attach to shared memory segments
    shm_in = shared_memory.SharedMemory(name=in_shm_name)
    shm_gray = shared_memory.SharedMemory(name=out_gray_name)
    shm_blur = shared_memory.SharedMemory(name=out_blur_name)
    shm_edges = shared_memory.SharedMemory(name=out_edges_name)

    try:
        # 2. Create zero-copy NumPy array views directly over shared memory buffers
        in_arr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm_in.buf, offset=in_offset)
        out_gray = np.ndarray((h, w), dtype=np.uint8, buffer=shm_gray.buf, offset=out_offset)
        out_blur = np.ndarray((h, w), dtype=np.uint8, buffer=shm_blur.buf, offset=out_offset)
        out_edges = np.ndarray((h, w), dtype=np.uint8, buffer=shm_edges.buf, offset=out_offset)

        # 3. Pipeline Step 1: Grayscale Conversion
        gray = to_grayscale(in_arr)
        out_gray[:] = gray[:]

        # 4. Pipeline Step 2: Separable Gaussian Blur
        blur = gaussian_blur(gray, kernel_size=kernel_size, sigma=sigma)
        out_blur[:] = blur[:]

        # 5. Pipeline Step 3: Sobel Edge Detection
        edges = sobel_edges(blur)
        out_edges[:] = edges[:]

        elapsed = time.perf_counter() - t_start

        return {
            "task_id": task_id,
            "name": name,
            "time_seconds": elapsed,
            "pid": os.getpid(),
        }

    finally:
        # Close local descriptors in worker (do NOT unlink from worker)
        shm_in.close()
        shm_gray.close()
        shm_blur.close()
        shm_edges.close()


@contextmanager
def shared_memory_batch_scope(
    images: List[Tuple[str, np.ndarray]],
    params: Dict[str, Any],
):
    """
    Context manager for deterministic allocation, writing, and cleanup
    of shared memory buffers for batch image processing.

    Yields:
        Tuple[List[Tuple], Dict[str, shared_memory.SharedMemory]]:
            (task_list, shm_dict)
    """
    if not images:
        raise ValueError("Cannot create shared memory batch for empty images list.")

    kernel_size = int(params.get("kernel_size", 15))
    sigma = float(params.get("sigma", 3.0))

    total_in_bytes = sum(arr.nbytes for _, arr in images)
    total_out_bytes = sum(arr.shape[0] * arr.shape[1] for _, arr in images)

    shm_in = None
    shm_gray = None
    shm_blur = None
    shm_edges = None

    try:
        shm_in = shared_memory.SharedMemory(create=True, size=total_in_bytes)
        shm_gray = shared_memory.SharedMemory(create=True, size=total_out_bytes)
        shm_blur = shared_memory.SharedMemory(create=True, size=total_out_bytes)
        shm_edges = shared_memory.SharedMemory(create=True, size=total_out_bytes)

        task_list = []
        in_offset = 0
        out_offset = 0

        for idx, (name, arr) in enumerate(images):
            h, w = arr.shape[:2]
            n_in = arr.nbytes
            n_out = h * w

            # Populate input shared memory buffer
            dest_view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm_in.buf, offset=in_offset)
            dest_view[:] = arr[:]

            task_list.append(
                (
                    idx,
                    name,
                    h,
                    w,
                    shm_in.name,
                    in_offset,
                    shm_gray.name,
                    shm_blur.name,
                    shm_edges.name,
                    out_offset,
                    kernel_size,
                    sigma,
                )
            )

            in_offset += n_in
            out_offset += n_out

        shm_dict = {
            "in": shm_in,
            "gray": shm_gray,
            "blur": shm_blur,
            "edges": shm_edges,
        }

        yield task_list, shm_dict

    finally:
        for shm in [shm_in, shm_gray, shm_blur, shm_edges]:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass


class PersistentWorkerPool:
    """
    Reusable multi-process worker pool for steady-state HPC execution.

    Maintains long-running worker processes across multiple benchmark trials
    and image batches, completely eliminating repeated process spawn overhead.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.ctx = mp.get_context("spawn")
        self._pool: Optional[mp.pool.Pool] = None
        self._is_alive = False
        self._init_duration: float = 0.0

    @property
    def is_alive(self) -> bool:
        """Return True if worker pool is initialized and active."""
        return self._is_alive

    def start(self) -> float:
        """
        Initialize the persistent worker pool and record startup latency.

        Returns:
            float: Initialization duration in seconds.
        """
        if self._is_alive and self._pool is not None:
            return 0.0

        t0 = time.perf_counter()
        self._pool = self.ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
        )
        self._init_duration = time.perf_counter() - t0
        self._is_alive = True
        return self._init_duration

    def warmup(self, sample_task: Tuple[Any, ...]) -> None:
        """Execute one untimed task on every worker to warm Python interpreter caches."""
        if not self._is_alive or self._pool is None:
            self.start()
        warmup_tasks = [sample_task] * self.num_workers
        self._pool.map(_worker_shm_task, warmup_tasks, chunksize=1)

    def execute_tasks(
        self,
        task_list: List[Tuple[Any, ...]],
        chunksize: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute tasks on the active pool with adaptive chunking."""
        if not self._is_alive or self._pool is None:
            self.start()

        if chunksize is None or chunksize < 1:
            chunksize = max(1, len(task_list) // (self.num_workers * 2))

        return self._pool.map(_worker_shm_task, task_list, chunksize=chunksize)

    def shutdown(self) -> None:
        """Terminate and clean up pool processes."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                self._pool.terminate()
            finally:
                self._pool = None
                self._is_alive = False

    @property
    def initialization_time_s(self) -> float:
        """Return the one-time process pool startup duration in seconds."""
        return self._init_duration


def process_images_parallel_optimized(
    images: List[Tuple[str, np.ndarray]],
    params: Dict[str, Any],
    num_workers: int,
    persistent_pool: Optional[PersistentWorkerPool] = None,
    chunksize: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Process multiple images in parallel using zero-copy shared memory and adaptive chunking.

    Args:
        images: List of (name, rgb_array) pairs.
        params: Filter settings dictionary ('kernel_size', 'sigma').
        num_workers: Number of worker processes to allocate.
        persistent_pool: Optional active PersistentWorkerPool instance for steady-state evaluation.
        chunksize: Optional task chunking size (auto-computed if None).

    Returns:
        Tuple[List[Dict[str, Any]], float]: Processed image results and measured wall-clock duration.
    """
    if not images:
        raise ValueError("The images list cannot be empty.")

    max_cpus = os.cpu_count() or 1
    if not isinstance(num_workers, int) or num_workers < 1 or num_workers > max_cpus:
        raise ValueError(
            f"num_workers must be an integer between 1 and {max_cpus}, got {num_workers}."
        )

    # Use existing persistent pool or create an ephemeral pool context
    pool_to_use = persistent_pool
    ephemeral_pool = None

    if pool_to_use is None:
        ephemeral_pool = PersistentWorkerPool(num_workers=num_workers)
        ephemeral_pool.start()
        pool_to_use = ephemeral_pool

    try:
        with shared_memory_batch_scope(images, params) as (task_list, shm_dict):
            t_start = time.perf_counter()

            raw_results = pool_to_use.execute_tasks(task_list, chunksize=chunksize)

            # Sort raw results by original task ordering
            raw_results.sort(key=lambda r: r["task_id"])

            # Reconstruct complete result dictionaries with local copies from shared memory
            results: List[Dict[str, Any]] = []
            out_offset = 0

            shm_gray = shm_dict["gray"]
            shm_blur = shm_dict["blur"]
            shm_edges = shm_dict["edges"]

            for meta, (name, arr) in zip(raw_results, images):
                h, w = arr.shape[:2]
                n_out = h * w

                # Zero-copy view into output shared memory
                v_gray = np.ndarray((h, w), dtype=np.uint8, buffer=shm_gray.buf, offset=out_offset)
                v_blur = np.ndarray((h, w), dtype=np.uint8, buffer=shm_blur.buf, offset=out_offset)
                v_edges = np.ndarray((h, w), dtype=np.uint8, buffer=shm_edges.buf, offset=out_offset)

                results.append(
                    {
                        "name": name,
                        "gray": v_gray.copy(),
                        "blur": v_blur.copy(),
                        "edges": v_edges.copy(),
                        "time_seconds": meta["time_seconds"],
                        "pid": meta["pid"],
                    }
                )
                out_offset += n_out

            total_seconds = time.perf_counter() - t_start

        return results, total_seconds

    finally:
        if ephemeral_pool is not None:
            ephemeral_pool.shutdown()
