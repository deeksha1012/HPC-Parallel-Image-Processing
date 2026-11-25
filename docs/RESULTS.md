# Benchmark Results and Performance Analysis

This document records the empirical performance benchmarks measured on the student's physical hardware.

> [!IMPORTANT]
> **No Fake Numbers**: All values below must be copied directly from your own runs saved in `results/benchmark_results.csv` and `results/system_info.json`. Do not invent or estimate timings.

---

## 1. Test Machine Details

*Copy these specifications from your generated `results/system_info.json` file:*

- **Timestamp of Benchmark:** FILL FROM YOUR OWN RUN
- **Operating System / Platform:** FILL FROM YOUR OWN RUN
- **Processor Architecture:** FILL FROM YOUR OWN RUN
- **Python Version:** FILL FROM YOUR OWN RUN
- **Total CPU Cores (os.cpu_count()):** FILL FROM YOUR OWN RUN
- **Process Start Method:** FILL FROM YOUR OWN RUN

---

## 2. Benchmark Parameters

*Record the exact parameter settings configured for your benchmark run:*

- **Number of Images:** FILL FROM YOUR OWN RUN
- **Image Resolution:** FILL FROM YOUR OWN RUN
- **Total Workload (Megapixels):** FILL FROM YOUR OWN RUN
- **Gaussian Kernel Size:** FILL FROM YOUR OWN RUN
- **Gaussian Sigma:** FILL FROM YOUR OWN RUN
- **Benchmark Repeats per Test:** FILL FROM YOUR OWN RUN

---

## 3. Measured Performance Results

*Copy these exact values from your generated `results/benchmark_results.csv` file:*

| Workers ($p$) | Mode | Mean Time (s) | Min Time (s) | Speedup ($S_p$) | Efficiency ($E_p$ %) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Sequential | | | 1.0000 | 100.0% |
| 2 | Parallel | | | | |
| 4 | Parallel | | | | |
| 8 | Parallel | | | | |
| 10 | Parallel | | | | |

---

## 4. Student Analysis and Observations

*Answer these three questions in your own words based on your measured numbers and generated charts:*

1. **Optimal Worker Configuration:**
   - *Which worker count achieved the lowest mean execution time on your machine? Did speedup continue to scale linearly as you added more cores, or did it plateau?*
   - **Student Response:** FILL FROM YOUR OWN RUN

2. **Parallel Efficiency Behavior:**
   - *How did parallel efficiency change as the number of worker cores increased? What hardware and software factors (e.g., process spawning overhead, IPC data transfer, Apple Silicon P-cores vs. E-cores) explain this behavior?*
   - **Student Response:** FILL FROM YOUR OWN RUN

3. **Comparison Against Sequential Baseline:**
   - *Did your 2-worker parallel run outperform the sequential baseline? Why is parallel processing sometimes slower than sequential execution for small image counts?*
   - **Student Response:** FILL FROM YOUR OWN RUN
