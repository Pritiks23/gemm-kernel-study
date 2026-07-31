!(https://rocm.blogs.amd.com/_images/software-tools-optimization-matrix-cores-cdna-images-matrix-cores-cdna.webp)


# GEMM Kernel Optimization Study

A CUDA benchmarking experiment analyzing how different matrix multiplication implementations affect GPU performance.

## Research Question

How does the choice of GEMM implementation impact execution time and GPU efficiency?

## Experimental Variable

The independent variable is:

**Matrix multiplication implementation strategy**

The following approaches are compared:

| Strategy | Description |
|---|---|
| CPU NumPy | CPU baseline implementation |
| cuBLAS GEMM | NVIDIA optimized FP32 GPU implementation |
| Tensor Core FP16 GEMM | FP16 matrix multiplication using Tensor Core acceleration |
| Naive CUDA Kernel | Custom CUDA implementation using global memory |
| Tiled CUDA Kernel | Custom CUDA implementation using shared memory reuse |

---

## Workload

Operation:
C = A × B


Matrix size:

All strategies use the same input matrices to ensure a controlled comparison.

---

## Metrics Measured

Each implementation is evaluated using:

- Runtime (milliseconds)
- GFLOPS
- GPU performance characteristics

---

## Experiment Design

```text
                 GEMM Benchmark
                        |
    ------------------------------------------------
    |                    |                         |
    v                    v                         v

CPU NumPy           GPU Implementations        Tensor Cores
Baseline                  |                    FP16 GEMM
                          |
                ---------------------
                |                   |
                v                   v

          Naive CUDA          Tiled CUDA
          Global Memory       Shared Memory
```

## Technologies

- CUDA C++
- PyTorch CUDA Extensions
- NVIDIA cuBLAS
- Tensor Cores
- Shared Memory Optimization

---

## Expected Findings

- CPU NumPy provides the baseline performance.
- CUDA kernels accelerate computation through GPU parallelism.
- Naive CUDA kernels are limited by repeated global memory access.
- Tiled CUDA improves performance by increasing data reuse through shared memory.
- cuBLAS achieves high performance through NVIDIA-optimized GEMM algorithms.
- Tensor Core FP16 execution provides the highest throughput by using specialized matrix acceleration hardware.

---


## Goal

This project studies how GPU performance changes when modifying the execution strategy of the same mathematical workload.

The goal is to understand the impact of:

- GPU parallelism
- memory reuse
- CUDA kernel design
- optimized GPU libraries
- specialized AI hardware acceleration

rather than only measuring which implementation is fastest.


4096 x 4096
