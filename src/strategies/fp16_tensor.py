import torch
import time


def run(A, B):
    """
    Runs GEMM using FP16 precision.
    On supported NVIDIA GPUs this uses Tensor Cores.
    """

    # Convert input matrices to FP16 CUDA tensors
    A_gpu = torch.from_numpy(A).half().cuda()
    B_gpu = torch.from_numpy(B).half().cuda()


    # Warmup
    C = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()


    start = time.perf_counter()


    C = torch.matmul(A_gpu, B_gpu)


    torch.cuda.synchronize()


    end = time.perf_counter()


    runtime_ms = (end - start) * 1000


    return runtime_ms
