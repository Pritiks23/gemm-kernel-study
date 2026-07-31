import numpy as np
import time


def run(A, B):

    start = time.perf_counter()

    C = A @ B

    end = time.perf_counter()

    runtime_ms = (end - start) * 1000

    return runtime_ms
