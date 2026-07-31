import torch
import numpy as np
import pandas as pd
import os

from strategies.cpu_numpy import run as cpu_numpy
from strategies.torch_cublas import run as torch_cublas


# Matrix size
N = 4096


def create_inputs():
    """
    Creates identical input matrices
    for every experiment.
    """

    np.random.seed(0)

    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    return A, B



def main():

    A, B = create_inputs()


    experiments = {
    "CPU NumPy": cpu_numpy,
    "cuBLAS GEMM": torch_cublas,
}


    results = []


    for name, experiment in experiments.items():

        print(f"Running {name}...")

        runtime = experiment(A, B)

        results.append(
            {
                "strategy": name,
                "runtime_ms": runtime
            }
        )


    df = pd.DataFrame(results)


    os.makedirs(
        "results",
        exist_ok=True
    )


    df.to_csv(
        "results/results.csv",
        index=False
    )


    print("\nResults:")
    print(df)



if __name__ == "__main__":
    main()
