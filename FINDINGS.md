
# GEMM Kernel Optimization Study — Findings

## Overview

This project benchmarks different GEMM (General Matrix Multiplication) implementations on an NVIDIA GPU to understand the performance impact of different execution strategies.

The experiment keeps the following factors constant:

- Matrix size: **4096 × 4096**
- Input data type: **FP32**
- Same input matrices generated with a fixed random seed
- Same GEMM operation: `C = A × B`
- Same hardware environment

The only variable changed is the GEMM implementation strategy.

The implementations compared:

1. CPU NumPy baseline
2. PyTorch CUDA GEMM (cuBLAS-backed GPU implementation)
3. Naive custom CUDA kernel
4. Tiled custom CUDA kernel using shared memory optimization

---

# Hardware Environment

GPU:

- NVIDIA GeForce RTX 4090

CUDA Environment:

- CUDA Version: 13.0
- Driver Version: 580.142

---

# Benchmark Results

| Strategy | Runtime (ms) | GFLOPS |
|---|---:|---:|
| CPU NumPy | 221.75 ms | 619.78 |
| PyTorch CUDA GEMM | 2.98 ms | 46,173.59 |
| Naive CUDA Kernel | 38.97 ms | 3,526.38 |
| Tiled CUDA Kernel | 21.18 ms | 6,489.43 |

---

# Key Findings

## 1. GPU Acceleration Provides a Large Performance Advantage

The PyTorch CUDA GEMM implementation achieved:

- Runtime: **2.98 ms**
- Throughput: **46.17 TFLOPS**

Compared to CPU NumPy:

\[
\frac{221.75}{2.98} \approx 74.5\times
\]

The GPU implementation is approximately **74× faster** than the CPU baseline.

This demonstrates the massive parallelism advantage of GPUs for dense matrix multiplication workloads.

---

# 2. Shared Memory Tiling Improves Custom CUDA Kernel Performance

The naive CUDA kernel achieved:

- Runtime: **38.97 ms**
- Throughput: **3.53 TFLOPS**

The tiled CUDA kernel achieved:

- Runtime: **21.18 ms**
- Throughput: **6.49 TFLOPS**

Performance improvement:

\[
\frac{38.97 - 21.18}{38.97} \times 100
\]

≈ **45.6% faster**

The tiled implementation nearly doubles throughput by improving memory access efficiency.

The improvement comes from:

- Loading matrix tiles into shared memory
- Reusing data across multiple operations
- Reducing repeated global memory accesses
- Improving memory locality

---

# 3. Custom CUDA Kernels Still Trail Optimized Libraries

Although tiling significantly improves the custom CUDA implementation, PyTorch CUDA GEMM remains much faster.

Comparison:

| Implementation | GFLOPS |
|-|-:|
| PyTorch CUDA GEMM | 46,173 GFLOPS |
| Tiled CUDA | 6,489 GFLOPS |

PyTorch CUDA GEMM achieves approximately:

\[
\frac{46173}{6489} \approx 7.1\times
\]

higher throughput.

This gap is expected because production GEMM libraries use many additional optimizations:

- Highly optimized kernel selection
- Architecture-specific tuning
- Tensor Core utilization when applicable
- Warp-level optimizations
- Register blocking
- Double buffering
- Extensive kernel auto-tuning

---

# 4. Memory Optimization Is Critical for GPU Performance

The custom kernel progression shows the importance of optimizing memory behavior:

Naive CUDA:

- Each thread repeatedly accesses global memory
- Poor data reuse
- Lower arithmetic intensity

Tiled CUDA:

- Data is staged into shared memory
- Threads reuse loaded values
- Global memory traffic is reduced

The result:

- Runtime decreased from **38.97 ms → 21.18 ms**
- Throughput increased from **3.53 TFLOPS → 6.49 TFLOPS**

---

# Conclusion

This experiment demonstrates the performance hierarchy of GEMM implementations:

```

CPU NumPy
↓
Naive CUDA Kernel
↓
Tiled CUDA Kernel
↓
Optimized GPU Libraries (PyTorch/cuBLAS)

```

The main takeaway is that GPU performance is not only determined by parallel execution, but by how efficiently computation is mapped onto GPU hardware.

The custom tiled CUDA kernel achieved a significant improvement over the naive implementation, showing the importance of:

- Shared memory usage
- Memory reuse
- Data locality
- GPU-aware algorithm design

However, optimized libraries such as cuBLAS remain substantially faster due to years of hardware-specific optimization and advanced kernel engineering.
