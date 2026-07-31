import torch
import time
import sys
import os


# Allow importing compiled extension
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../build"
    )
)


import tiled_cuda



def run(A, B):
    """
    Runs custom tiled CUDA GEMM kernel.
    """

    # Convert numpy -> CUDA tensors
    A_gpu = torch.from_numpy(A).cuda()
    B_gpu = torch.from_numpy(B).cuda()


    torch.cuda.synchronize()


    start = time.perf_counter()


    C = tiled_cuda.tiled_gemm(
        A_gpu,
        B_gpu
    )


    torch.cuda.synchronize()


    end = time.perf_counter()


    runtime_ms = (end - start) * 1000


    return runtime_ms
