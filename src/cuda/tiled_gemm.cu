#include <torch/extension.h>

#define TILE_SIZE 16

__global__ void tiled_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
)
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Number of tiles needed
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++)
    {
        int tiled_col = tile * TILE_SIZE + threadIdx.x;
        int tiled_row = tile * TILE_SIZE + threadIdx.y;

        // Load A tile safely
        if (row < N && tiled_col < N)
        {
            tile_A[threadIdx.y][threadIdx.x] =
                A[row * N + tiled_col];
        }
        else
        {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile safely
        if (tiled_row < N && col < N)
        {
            tile_B[threadIdx.y][threadIdx.x] =
                B[tiled_row * N + col];
        }
        else
        {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            value +=
                tile_A[threadIdx.y][k] *
                tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store result safely
    if (row < N && col < N)
    {
        C[row * N + col] = value;
    }
}

torch::Tensor tiled_gemm(
    torch::Tensor A,
    torch::Tensor B
)
{
    int N = A.size(0);

    auto C = torch::zeros(
        {N, N},
        A.options()
    );

    dim3 threads(
        TILE_SIZE,
        TILE_SIZE
    );

    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    tiled_gemm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
)
{
    m.def(
        "tiled_gemm",
        &tiled_gemm,
        "Tiled CUDA GEMM"
    );
}
