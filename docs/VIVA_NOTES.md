# Viva Voce Examination Notes & Demo Script

---

## 1. 30-Second Elevator Pitch

"This project evaluates High Performance Computing techniques applied to digital image processing using Python. We built pure NumPy implementations of grayscale conversion, 2-pass separable Gaussian blur, and Sobel edge detection to avoid black-box library abstractions. We then built a data-parallel batch processing pipeline using Python's standard `multiprocessing` library under the macOS `spawn` architecture. Finally, we developed an interactive Streamlit dashboard that measures execution time across CPU core allocations, computes parallel speedup and efficiency, visualizes scaling curves, and exports processed images."

---

## 2. 3-Minute Live Demonstration Script

1. **Launch the Dashboard (0:00 - 0:30)**:
   - Run `streamlit run app.py` in the terminal.
   - Point out the clean layout: configuration controls in the sidebar, main area presenting project goals, and an explicit note on when parallel processing is advantageous.
2. **Configure Parameters (0:30 - 1:00)**:
   - Upload sample images from `sample_images/` using the file uploader.
   - Show how image dimensions and total workload in megapixels are automatically validated.
   - Select Gaussian kernel size (e.g., 15) and worker allocations (e.g., 2, 4, and max CPU cores).
3. **Execute Benchmark (1:00 - 1:45)**:
   - Click **Run benchmark**.
   - Show the dynamic progress bar updating across sequential baseline and parallel trials.
   - Explain that an untimed warm-up pass precedes sequential timing, while parallel timing deliberately encompasses the entire process pool lifecycle.
4. **Analyze Performance Metrics & Charts (1:45 - 2:30)**:
   - Review the summary metric cards: Sequential Baseline, Fastest Parallel configuration, Maximum Speedup, and Parallel Efficiency.
   - Inspect the **Benchmark Metrics Table**.
   - Switch to the **Performance Graphs** tabs:
     - *Execution Time*: Show the bar chart and standard deviation error bars.
     - *Speedup*: Compare measured speedup against the theoretical linear reference line ($S = p$).
     - *Efficiency*: Highlight the downward trend illustrating Amdahl's Law and IPC communication friction.
5. **Inspect & Export Outputs (2:30 - 3:00)**:
   - Navigate to the **Processed Images Preview** selectbox and display the 4-column side-by-side comparison: Original RGB, Grayscale, Gaussian Blur, and Sobel Edges.
   - Demonstrate the **Downloads** section: download individual stage PNGs, download high-res chart PNGs, download the CSV data table, and generate an in-memory ZIP archive of processed outputs.

---

## 3. Top 15 Viva Voce Questions & Model Answers

### Q1: What is parallel computing?
**Answer**: Parallel computing is a computational technique where multiple processing units (CPU cores or nodes) execute independent calculations simultaneously to solve a problem faster than a sequential, single-core approach.

### Q2: What is the difference between data parallelism and task parallelism?
**Answer**: In **data parallelism**, the same computational algorithm is applied concurrently across distinct subsets of data (such as processing 12 different images simultaneously across cores). In **task parallelism**, distinct, different algorithms or tasks run concurrently on the same or different data.

### Q3: Why did we use Python `multiprocessing` instead of `threading` for CPU-bound image processing?
**Answer**: Standard CPython features a Global Interpreter Lock (GIL), which permits only one thread to execute Python bytecode at any given moment. For CPU-bound tasks like image filtering, multi-threading cannot achieve true parallel execution across multiple cores. The `multiprocessing` library bypasses the GIL by spawning independent Python processes, each with its own interpreter and memory space.

### Q4: What is the difference between `spawn` and `fork` start methods on macOS?
**Answer**: `fork` duplicates the parent process memory, which is unsafe on macOS because system frameworks (CoreFoundation, Metal) can deadlock in child processes. `spawn` starts an entirely clean, new Python interpreter process that safely re-imports modules from disk, ensuring thread-safe execution on Apple Silicon.

### Q5: Why is the `if __name__ == "__main__":` guard strictly required?
**Answer**: Under the `spawn` method, child worker processes re-import the main script from scratch. Without the `__main__` guard, each spawned child process would re-execute the pool creation code, creating an infinite, recursive cascade of processes that crashes the operating system.

### Q6: What is pickling, and why does it introduce parallel overhead?
**Answer**: Pickling is Python's mechanism for serializing in-memory objects into byte streams so they can be transmitted across inter-process communication (IPC) pipes. Pickling large uncompressed NumPy arrays takes CPU time and memory bandwidth, adding overhead that does not exist in single-threaded sequential execution.

### Q7: What are the mathematical definitions of Speedup and Parallel Efficiency?
**Answer**:
- **Speedup ($S_p$)**: $S_p = \frac{T_1}{T_p}$, where $T_1$ is sequential execution time and $T_p$ is parallel execution time on $p$ cores.
- **Parallel Efficiency ($E_p$)**: $E_p = \frac{S_p}{p} = \frac{T_1}{p \cdot T_p}$. It measures the fraction of worker core capacity converted into speedup.

### Q8: What is Amdahl's Law, and how does it apply here?
**Answer**: Amdahl's Law states that the maximum achievable speedup is bounded by the sequential fraction of the program ($s$):
$$\text{Speedup}_{\max} = \frac{1}{s + \frac{1 - s}{p}}$$
In our application, operations like initial file reading, pool creation, argument pickling, and result aggregation cannot be parallelized, imposing an upper limit on speedup regardless of core count.

### Q9: Why can parallel speedup sometimes be less than 1.0 (slower than sequential)?
**Answer**: When working with few images or small image resolutions, the total compute workload is very small. The overhead of spawning worker processes, transmitting data through IPC pipes, and cleaning up the pool exceeds the time saved by parallel execution.

### Q10: Why does parallel efficiency typically decline as more worker cores are added?
**Answer**: As worker count increases, contention for memory bus bandwidth grows, IPC transmission overhead multiplies, and Amdahl's sequential fraction occupies a larger percentage of total execution time. Additionally, worker scheduling desynchronization causes cores to wait idle for the slowest process to finish.

### Q11: How does Apple Silicon hardware architecture impact parallel benchmarks?
**Answer**: Apple Silicon (M-series) utilizes a heterogeneous architecture featuring high-performance cores (P-cores) and high-efficiency cores (E-cores). When the worker count exceeds the number of P-cores, tasks assigned to E-cores execute at lower clock frequencies, creating load imbalances where faster P-cores must wait for slower E-cores to complete.

### Q12: Why did we implement filters directly in NumPy instead of using OpenCV or SciPy?
**Answer**: Using NumPy ensures that every line of image processing logic (luminance math, 1D separable convolution slicing, and 3x3 Sobel kernel gradients) is transparent and explainable during examination, demonstrating core algorithmic understanding rather than relying on compiled third-party C++ libraries.

### Q13: How is algorithmic correctness tested in this project?
**Answer**: Our automated Pytest suite tests filters against known analytical solutions: pure color luminance values ($R \to 76$, $G \to 150$, $B \to 29$), Gaussian kernel symmetry and sum normalization ($= 1.0$), variance reduction on noise, and Sobel edge detection across step functions. It also proves that sequential and parallel pipelines yield mathematically identical NumPy pixel arrays (`np.array_equal`).

### Q14: How are benchmark timings made scientifically reliable?
**Answer**:
1. We run an untimed sequential warm-up pass to eliminate library and memory allocation bias.
2. We repeat trials multiple times and report mean and standard deviation.
3. We log hardware, OS, and Python specifications alongside results.
4. We deliberately include pool creation and IPC in parallel timing to measure honest wall-clock user experience.

### Q15: What are the primary future improvements for this project?
**Answer**:
1. **Shared Memory (`multiprocessing.shared_memory`)**: Share NumPy array memory directly between processes without pickling or IPC pipe transmission.
2. **Spatial Domain Tiling**: Split a single massive image into overlapping sub-grid tiles to parallelize individual high-resolution images.
3. **GPU Acceleration**: Implement kernel operations using Metal (via PyTorch/MLX) or CUDA to evaluate SIMD parallel architectures against multi-core CPU multiprocessing.
