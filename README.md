![](https://rocm.blogs.amd.com/_images/software-tools-optimization-matrix-cores-cdna-images-matrix-cores-cdna.webp)

# GEMM Kernel Optimization Study

A CUDA benchmarking experiment analyzing how different matrix multiplication implementations affect GPU performance.

## Research Question

How does the choice of GEMM implementation impact execution time and GPU throughput when solving the same matrix multiplication workload?

---

# Experimental Variable

The independent variable is:

**Matrix multiplication implementation strategy**

All implementations compute the same operation:

\[
C = A x B
\]

while changing only the execution strategy.

The following approaches are compared:

---

# 🟢 GEMM Computation Strategies

| 🧩 Strategy | 📝 Description |
|---|---|
| 🖥️ **CPU NumPy** | CPU baseline implementation using NumPy |
| 🟩 **PyTorch CUDA GEMM** | GPU matrix multiplication using PyTorch CUDA backend (optimized GPU library path) |
| 🔵 **Naive CUDA Kernel** | Custom CUDA kernel where each thread computes output elements using global memory accesses |
| 🟠 **Tiled CUDA Kernel** | Custom CUDA kernel using shared memory tiling to improve data reuse and reduce global memory traffic |

---

# Workload

Operation:

\[
C = A x B
\]

Matrix dimensions:

```

4096 x 4096

````

Input properties:

- FP32 precision
- Identical input matrices across all experiments
- Same mathematical workload for every implementation

The only variable changed is the GEMM implementation strategy.

---

# Metrics Measured

Each implementation is evaluated using:

- Runtime (milliseconds)
- GFLOPS (billions of floating point operations per second)

GFLOPS calculation:

$$
GFLOPS = \frac{2N^3}{runtime(ms)\times10^6}
$$

where:

- \(N = 4096\)
- \(2N^3\) represents GEMM floating point operations

---

# Results

| Strategy | Runtime (ms) | GFLOPS |
|---|---:|---:|
| CPU NumPy | 221.75 | 619.78 |
| PyTorch CUDA GEMM | 2.98 | 46,173.59 |
| Naive CUDA Kernel | 38.97 | 3,526.38 |
| Tiled CUDA Kernel | 21.18 | 6,489.43 |

---

# Performance Analysis

## GPU Acceleration vs CPU Baseline

Compared to CPU NumPy:

$$
\frac{221.75}{2.98} \approx 74.5\times
$$

PyTorch CUDA GEMM achieved approximately **74.5x faster execution** than the CPU implementation.

This demonstrates the advantage of massively parallel GPU execution for large matrix operations.

---

## Shared Memory Optimization

The tiled CUDA kernel improved performance over the naive CUDA implementation.

Performance improvement:

$$
\frac{38.97 - 21.18}{38.97} \times 100 \approx 45.6\%
$$

≈ **45.6% improvement**

By loading tiles into shared memory, the kernel reduces redundant global memory accesses and increases data reuse.

---

## Optimized Library Performance

Compared with the custom tiled CUDA kernel:

$$
\frac{46173}{6489} \approx 7.1\times
$$

PyTorch CUDA GEMM achieved approximately **7.1x higher throughput**.

This highlights the performance advantage of highly optimized GPU libraries that incorporate:

- architecture-specific tuning
- optimized memory movement
- kernel selection strategies
- advanced GPU scheduling

---

# Experiment Design

```text
                    GEMM Benchmark
                           |
        ----------------------------------------
        |                  |                   |
        v                  v                   v

    CPU Baseline      GPU Library        Custom CUDA Kernels
        |                  |                   |
        |                  |          --------------------
        |                  |          |                  |
        v                  v          v                  v

   CPU NumPy       PyTorch CUDA   Naive CUDA       Tiled CUDA
                                  Global Memory    Shared Memory
````

---

# Technologies

* CUDA C++
* CUDA Runtime API
* PyTorch CUDA Extensions
* NVIDIA GPU Architecture
* CUDA Shared Memory
* GPU Performance Benchmarking

---

# Key Findings

* GPU acceleration dramatically improves GEMM performance compared to CPU execution.
* PyTorch CUDA GEMM provides the highest throughput due to highly optimized GPU kernels.
* Naive CUDA kernels are limited by inefficient global memory access patterns.
* Tiled CUDA improves performance by increasing memory reuse through shared memory.
* Custom CUDA kernels provide insight into GPU optimization techniques, while optimized libraries demonstrate the performance possible with production-level implementations.

---

# Goal

This project studies how GPU performance changes when modifying the execution strategy of the same mathematical workload.

The goal is to understand the impact of:

* GPU parallelism
* memory hierarchy
* shared memory optimization
* CUDA kernel design
* optimized GPU libraries

rather than only measuring which implementation is fastest.

```



