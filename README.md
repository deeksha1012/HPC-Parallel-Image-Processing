# High Performance Image Processing Using Parallel Computing

A High Performance Computing (HPC) mini project evaluating sequential versus data-parallel image processing across multi-core CPUs.
The application performs grayscale conversion, Gaussian blur filtering, and Sobel edge detection across batches of images using pure NumPy and Python's standard `multiprocessing` library.
It systematically evaluates execution times, speedup factors, and parallel efficiency across varying worker core allocations through an interactive Streamlit web dashboard and command-line interfaces.

Status: All 7 modules complete

---

## Features

- **Pure NumPy Filter Pipeline**: Self-contained implementations of ITU-R BT.601 grayscale conversion, 2-pass separable 1D Gaussian blur convolution, and 3x3 Sobel gradient edge detection without external image processing libraries (OpenCV/SciPy).
- **Data-Parallel Multiprocessing**: Scalable batch execution leveraging Python's standard `multiprocessing` library using the macOS-safe `spawn` process start method.
- **Fair Benchmarking**: Controlled benchmarking protocol featuring an untimed sequential warm-up pass, multi-trial repetition, and end-to-end wall-clock timing that accounts for process creation and inter-process communication (IPC) overhead.
- **HPC Metric Evaluation**: Automated measurement of execution time, parallel speedup ($S_p = T_1 / T_p$), and parallel efficiency ($E_p = S_p / p$).
- **Visual Analytics**: Publication-ready Matplotlib performance charts comparing execution times (with standard deviation error bars), speedup scaling against ideal linear speedup, and efficiency curves against the 100% baseline.
- **Interactive Web Interface**: Streamlit dashboard supporting multi-file image uploads, filter parameter controls, live progress tracking, summary metric cards, 4-stage visual preview gallery, and in-memory export (PNGs, CSV, and ZIP archives).
- **Comprehensive Unit Testing**: Robust Pytest test suite validating filter mathematics, multiprocessing consistency, input validation, and in-memory export pipelines.

---

## Tech Stack

- **Language**: Python 3.12 (ARM64 Apple Silicon native)
- **Numerical Computation**: NumPy
- **Image Handling & I/O**: Pillow (PIL)
- **Data Structures & Metrics**: Pandas
- **Visualization**: Matplotlib (headless `Agg` backend)
- **Web User Interface**: Streamlit
- **Testing Framework**: Pytest
- **Parallel Computing**: Standard Library `multiprocessing` (Process Pool, `spawn` context)

---

## Folder Structure

```
HPC-Parallel-Image-Processing/
├── .gitignore                      # Git exclusion rules for caches, venvs, and generated outputs
├── AI_USAGE.md                     # Disclosure of AI assistance and student verification log
├── README.md                       # Comprehensive project documentation
├── app.py                          # Streamlit web dashboard
├── pytest.ini                      # Pytest test discovery and path configuration
├── requirements.txt                # Package dependencies with lower bounds
├── docs/
│   ├── .gitkeep                    # Directory preservation marker
│   ├── RESULTS.md                  # Template for recording student benchmark results
│   └── VIVA_NOTES.md               # Quick review notes, demo script, and viva Q&A
├── results/
│   ├── .gitkeep                    # Directory preservation marker
│   ├── benchmark_results.csv       # Measured benchmark metrics table
│   ├── system_info.json            # Hardware and runtime environment details
│   ├── plots/                      # Exported high-resolution benchmark charts
│   └── sequential_output/          # Exported sequential image outputs
├── sample_images/                  # Synthetic test images for benchmarks
├── scripts/
│   └── generate_sample_images.py  # Script generating reproducible synthetic test images
├── src/
│   ├── __init__.py                 # Package marker for src
│   ├── benchmark.py                # Automated benchmarking engine and CLI
│   ├── config.py                   # Central paths, constants, and default parameters
│   ├── export.py                   # In-memory byte serialization (PNG, CSV, ZIP)
│   ├── filters.py                  # Pure-NumPy grayscale, Gaussian blur, and Sobel filters
│   ├── image_io.py                 # Image loading, folder discovery, and saving helpers
│   ├── metrics.py                  # Speedup and efficiency calculation functions
│   ├── parallel.py                 # Data-parallel batch processor using worker pools
│   ├── pipeline.py                 # Picklable single-image transformation pipeline
│   ├── plots.py                    # Matplotlib chart generators and export CLI
│   └── sequential.py               # Single-core baseline batch processor and CLI
└── tests/
    ├── __init__.py                 # Package marker for tests
    ├── conftest.py                 # Shared pytest fixtures (synthetic images, real benchmark data)
    ├── test_benchmark.py           # Benchmark engine validation and error handling tests
    ├── test_export_and_plots.py    # In-memory export serialization and plot creation tests
    ├── test_filters.py             # Filter mathematical correctness and boundary tests
    ├── test_metrics.py             # Speedup and efficiency formula unit tests
    ├── test_pipeline.py            # Single-image pipeline and parallel consistency tests
    └── test_smoke.py               # Initial setup and configuration smoke test
```

---

## Setup on macOS (Apple Silicon)

Follow these steps in your terminal to configure the environment:

1. **Navigate to the project root:**
   ```bash
   cd ~/Documents/HPC-Parallel-Image-Processing
   ```

2. **Create a virtual environment (if not already present):**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Generate synthetic benchmark images:**
   ```bash
   python scripts/generate_sample_images.py --count 12 --width 2000 --height 1500
   ```

---

## How to Run

### 1. Interactive Streamlit Dashboard
Launch the web interface in your browser:
```bash
streamlit run app.py
```

### 2. Sequential Processing CLI
Run the single-core baseline pipeline on a folder of images:
```bash
python -m src.sequential --input sample_images --output results/sequential_output
```

### 3. Parallel Processing CLI
Run the data-parallel pipeline across worker processes with sequential verification:
```bash
python -m src.parallel --input sample_images --workers 4 --verify
```

### 4. Automated Benchmark Suite CLI
Benchmark sequential vs. multiple worker allocations, saving CSV and system info:
```bash
python -m src.benchmark --input sample_images --workers 2 4 8 --repeats 3
```

### 5. Performance Plotting CLI
Generate high-resolution PNG charts from benchmark results:
```bash
python -m src.plots --csv results/benchmark_results.csv --out results/plots
```

### 6. Automated Pytest Test Suite
Execute all unit, integration, and mathematical correctness tests:
```bash
pytest -v
```

---

## How It Works

### Processing Pipeline
Every image undergoes three sequential filtering operations:
1. **Grayscale Conversion**: Uses the standard ITU-R BT.601 perceptual luminance formula ($Y = 0.299R + 0.587G + 0.114B$) calculated in `float32`, rounded, clamped, and cast to `uint8`.
2. **Gaussian Blur**: Employs a separable 1D convolution. A 1D discrete Gaussian kernel ($G(x) \propto e^{-x^2 / 2\sigma^2}$) is applied horizontally across rows, then vertically down columns. This reduces computational complexity from $O(K^2)$ to $O(2K)$ operations per pixel.
3. **Sobel Edge Detection**: Applies $3 \times 3$ horizontal ($G_x$) and vertical ($G_y$) convolution masks using vectorized array slicing on the blurred image, computes Euclidean gradient magnitude ($\sqrt{G_x^2 + G_y^2}$), and normalizes the maximum intensity to 255.

### Data Parallelism
- **Domain Decomposition**: Each image in the batch represents an independent, embarrassingly parallel task.
- **Process Pool**: Tasks are distributed across worker processes using `multiprocessing.Pool.map`.

### Why `spawn` on macOS Apple Silicon
macOS prohibits multi-threaded applications from using `fork()` without `exec()`, as inherited locks and CoreFoundation state lead to deadlocks and crashes.
Using `multiprocessing.get_context("spawn")` guarantees that each worker starts as a clean, brand-new Python interpreter process that safely re-imports modules.

### What is Timed
To measure honest, real-world parallel performance:
- Sequential timing measures the loop over all images after an untimed warm-up pass.
- Parallel timing wraps the entire `with ctx.Pool(...) as pool:` block. This includes process creation latency, IPC argument pickling, worker execution, result return pickling, and pool teardown.

---

## Performance Formulas

1. **Speedup ($S_p$)**:
   $$\text{Speedup } (S_p) = \frac{T_1}{T_p}$$
   *Where $T_1$ is the mean sequential execution time, and $T_p$ is the mean parallel execution time using $p$ worker processes.*

2. **Parallel Efficiency ($E_p$)**:
   $$\text{Parallel Efficiency } (E_p) = \frac{S_p}{p}$$
   *Expressed as a fraction or percentage ($E_p \times 100\%$). An efficiency of 1.0 (100%) represents ideal linear scaling.*

---

## Results

Benchmark measurements vary across machines depending on CPU architecture, logical core count, thermal conditions, and image dimensions.

> [!NOTE]
> Detailed benchmark tables and performance observations are recorded directly from local runs.
> Please refer to [docs/RESULTS.md](file:///Users/sinchanavs/Documents/HPC-Parallel-Image-Processing/docs/RESULTS.md) to view the student's recorded results.

---

## Limitations

- **Small Workloads**: For small image counts or low resolutions, process creation latency and memory serialization overhead outweigh parallel computational benefits, resulting in $S_p < 1.0$.
- **Amdahl's Law**: The sequential portions of the application (initial file loading, argument packaging, and result aggregation) establish a theoretical upper bound on speedup.
- **Apple Silicon Core Heterogeneity**: M-series processors combine high-performance cores (P-cores) and high-efficiency cores (E-cores). When worker counts exceed the number of P-cores, tasks assigned to E-cores take longer to finish, causing worker desynchronization.
- **Memory Overhead**: IPC communication requires pickling and transmitting uncompressed NumPy image buffers through inter-process pipes, multiplying memory bandwidth pressure.
- **Per-Image Granularity**: Parallelism is implemented across images rather than within a single image. A single massive image cannot be split across multiple cores in this design.
- **No Hardware Acceleration**: Computation runs entirely on CPU cores without leveraging Metal, Apple Neural Engine (ANE), or GPU shaders.

---

## Troubleshooting

- **Python Architecture Verification**:
  Ensure Python runs natively as ARM64 on Apple Silicon:
  ```bash
  python3 -c "import platform; print(platform.machine())"
  # Expected output: arm64
  ```
- **Port Conflict (Streamlit)**:
  If port 8501 is occupied, launch on an alternative port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **Missing Dependencies**:
  Ensure packages are installed in the activated virtual environment:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

---

## Academic AI Assistance Disclosure

This project was built with the assistance of an AI coding agent (Google Antigravity) under student direction.
For complete review records, automated testing logs, and student declarations, see [AI_USAGE.md](file:///Users/sinchanavs/Documents/HPC-Parallel-Image-Processing/AI_USAGE.md).
