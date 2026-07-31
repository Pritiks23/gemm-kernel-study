import torch
import time


def run(A, B):
    """
    Runs matrix multiplication using
    PyTorch CUDA backend (cuBLAS).
    """

    # Convert NumPy arrays to CUDA tensors
    A_gpu = torch.from_numpy(A).cuda()
    B_gpu = torch.from_numpy(B).cuda()


    # Warmup
    C = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()


    start = time.perf_counter()


    C = torch.matmul(A_gpu, B_gpu)


    # GPU operations are asynchronous,
    # so wait before measuring
    torch.cuda.synchronize()


    end = time.perf_counter()


    runtime_ms = (end - start) * 1000


    return runtime_ms
