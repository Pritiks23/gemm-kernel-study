# GEMM Kernel Optimization Study - Findings

## Research Question

**How does the choice of GEMM implementation impact execution time and GPU efficiency?**

This experiment compares different matrix multiplication strategies to understand how hardware acceleration, optimized libraries, and CUDA kernel design affect GPU performance.

---

# Experimental Setup

## Hardware

- **GPU:** NVIDIA GeForce RTX 4090
- **CUDA Version:** 13.0
- **PyTorch CUDA:** CUDA-enabled build
- **Matrix Operation:** C = A × B
- **Precision Types:**
  - FP32 for CPU NumPy, cuBLAS, and custom CUDA kernels
  - FP16 for Tensor Core implementation

---

# Implementations Compared

| Strategy | Description |
|---|---|
| CPU NumPy | CPU baseline implementation using NumPy matrix multiplication |
| cuBLAS GEMM | NVIDIA optimized FP32 GPU matrix multiplication library |
| Tensor Core FP16 GEMM | FP16 matrix multiplication using NVIDIA Tensor Core acceleration |
| Naive CUDA Kernel | Custom CUDA kernel using direct global memory accesses |
| Tiled CUDA Kernel | Custom CUDA kernel using shared memory tiling and data reuse |

---

# Benchmark Results

| Implementation | Runtime (ms) |
|---|---:|
| CPU NumPy | 155.85 ms |
| cuBLAS GEMM | 2.94 ms |
| Tensor Core FP16 GEMM | 0.97 ms |
| Naive CUDA Kernel | 38.43 ms |
| Tiled CUDA Kernel | 20.98 ms |

---

# Performance Analysis

## 1. Tensor Core FP16 GEMM Achieved Highest Performance

The Tensor Core implementation achieved the fastest runtime:

```

0.966 ms

```

This performance improvement comes from NVIDIA Tensor Core hardware acceleration.

Tensor Cores are specialized GPU units designed for high-throughput matrix operations, particularly for lower precision formats such as FP16.

Compared with CPU NumPy:

\[
\frac{155.85}{0.966} \approx 161\times
\]

The Tensor Core implementation was approximately **161x faster than the CPU baseline**.

---

# 2. cuBLAS Provided Highly Optimized FP32 Performance

The cuBLAS GEMM implementation achieved:

```

2.94 ms

```

Even though it uses FP32 arithmetic, it significantly outperformed the custom CUDA kernels.

cuBLAS benefits from years of NVIDIA optimization, including:

- optimized thread scheduling
- register blocking
- shared memory utilization
- memory coalescing
- architecture-specific kernel selection

This demonstrates why production ML frameworks typically rely on highly optimized libraries rather than handwritten kernels.

---

# 3. Shared Memory Tiling Improved CUDA Kernel Performance

The custom CUDA kernels showed a clear difference between naive and tiled approaches.

## Naive CUDA Kernel

Runtime:

```

38.43 ms

```

The naive implementation performs repeated global memory accesses. Since global memory has high latency, threads spend significant time waiting for data.

---

## Tiled CUDA Kernel

Runtime:

```

20.98 ms

```

The tiled implementation improved performance by:

\[
\frac{38.43-20.98}{38.43}\times100
\]

≈ **45.4% speedup**

The improvement comes from loading matrix blocks into shared memory:

1. Threads cooperatively load tiles from global memory.
2. Data is reused by multiple threads.
3. Expensive global memory accesses are reduced.
4. Arithmetic intensity increases.

---

# Performance Ranking

Fastest to Slowest:

| Rank | Implementation | Runtime |
|---|---|---:|
| 1 | Tensor Core FP16 GEMM | 0.97 ms |
| 2 | cuBLAS GEMM | 2.94 ms |
| 3 | Tiled CUDA Kernel | 20.98 ms |
| 4 | Naive CUDA Kernel | 38.43 ms |
| 5 | CPU NumPy | 155.85 ms |

---

# Key Findings

## Finding 1: Hardware acceleration dominates GEMM performance

GPU implementations dramatically outperform CPU computation.

The Tensor Core implementation achieved over two orders of magnitude improvement compared with CPU NumPy.

---

## Finding 2: Memory optimization is critical for CUDA kernels

The tiled CUDA kernel demonstrates that reducing global memory traffic through shared memory reuse significantly improves performance.

CUDA performance is not only determined by computation throughput but also by efficient memory access patterns.

---

## Finding 3: Optimized libraries outperform simple custom kernels

Although custom CUDA kernels provide control over execution behavior, libraries such as cuBLAS contain highly optimized implementations that are difficult to match without extensive kernel tuning.

---

# Conclusion

This experiment demonstrates that GEMM performance depends heavily on implementation strategy.

The progression from CPU NumPy → custom CUDA → tiled CUDA → cuBLAS → Tensor Core GEMM shows the impact of:

- GPU parallelism
- memory hierarchy optimization
- specialized hardware acceleration
- optimized numerical libraries

The Tensor Core FP16 implementation achieved the highest performance, while shared-memory tiling provided a significant improvement over naive CUDA programming.

These results highlight the importance of kernel optimization techniques used in modern GPU-accelerated machine learning systems.
