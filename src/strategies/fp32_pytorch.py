import torch
import time


def run(A, B):
    """
    Runs GEMM using FP32 precision.
    """

    # Convert input matrices to FP16 CUDA tensors
    A_gpu = torch.from_numpy(A).float().cuda()
    B_gpu = torch.from_numpy(B).float().cuda()


    # Warmup
    C = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()


    start = time.perf_counter()


    C = torch.matmul(A_gpu, B_gpu)


    torch.cuda.synchronize()


    end = time.perf_counter()


    runtime_ms = (end - start) * 1000


    return runtime_ms
