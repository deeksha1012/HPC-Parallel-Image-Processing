# High Performance Image Processing Using Parallel Computing

This project investigates high-performance batch image processing using Python.
It applies grayscale conversion, Gaussian blur, and Sobel edge detection to multiple images sequentially and in parallel using Python multiprocessing.
The project evaluates computational performance metrics including execution time, speedup, and parallel efficiency across varying numbers of CPU cores.

Status: Module 5 of 7 (Streamlit dashboard)

## Setup Guide (macOS / Apple Silicon)

Follow these steps to set up and run the project environment:

1. **Navigate to the project root:**
   ```bash
   cd ~/Documents/HPC-Parallel-Image-Processing
   ```

2. **Create a virtual environment (if not already created):**
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

5. **Generate synthetic sample images:**
   ```bash
   python scripts/generate_sample_images.py
   ```

6. **Run smoke tests:**
   ```bash
   pytest
   ```

## Planned Features

- **Sequential Pipeline**: Implementation of pure-Python and NumPy image processing algorithms (RGB to grayscale, Gaussian blur convolution, and Sobel edge detection).
- **Parallel Processing Pipeline**: Multi-core batch image processing leveraging Python's `multiprocessing` library.
- **HPC Metric Evaluation**: Automated measurement and comparison of execution time, speedup ($S_p = T_1 / T_p$), and parallel efficiency ($E_p = S_p / p$).
- **Performance Visualizations**: Automated generation of comparative performance charts and scaling curves using Matplotlib and Pandas.
- **Interactive Streamlit Dashboard**: Web user interface allowing image uploads, parameter adjustments, live pipeline execution, performance metric comparison, and download of processed images.

## Project Structure

```
HPC-Parallel-Image-Processing/
├── .gitignore                      # Git ignore patterns for temporary files and caches
├── AI_USAGE.md                     # Disclosure and review log of AI assistance
├── README.md                       # Project overview, setup instructions, and documentation
├── pytest.ini                      # Pytest test discovery and path configuration
├── requirements.txt                # Project package dependencies
├── docs/                           # Project documentation, notes, and academic report drafts
├── results/                        # Generated output images and benchmark data (.gitkeep)
├── sample_images/                  # Synthetic test images for benchmarks (.gitkeep)
├── scripts/
│   └── generate_sample_images.py  # Script to generate synthetic test images
├── src/
│   ├── __init__.py                 # Package marker for src module
│   └── config.py                   # Central configuration constants and paths
└── tests/
    ├── __init__.py                 # Package marker for tests module
    └── test_smoke.py               # Smoke tests validating project setup
```
