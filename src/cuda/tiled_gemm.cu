#include <torch/extension.h>


#define TILE_SIZE 16



__global__ void tiled_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
){

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];


    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;


    float value = 0.0f;


    for(int tile = 0; tile < N / TILE_SIZE; tile++){

        tile_A[threadIdx.y][threadIdx.x] =
            A[row * N + tile * TILE_SIZE + threadIdx.x];


        tile_B[threadIdx.y][threadIdx.x] =
            B[(tile * TILE_SIZE + threadIdx.y) * N + col];


        __syncthreads();


        for(int k = 0; k < TILE_SIZE; k++){

            value +=
                tile_A[threadIdx.y][k] *
                tile_B[k][threadIdx.x];

        }


        __syncthreads();

    }


    C[row * N + col] = value;

}



torch::Tensor tiled_gemm(
    torch::Tensor A,
    torch::Tensor B
){

    int N = A.size(0);


    auto C = torch::zeros(
        {N,N},
        A.options()
    );


    dim3 threads(
        TILE_SIZE,
        TILE_SIZE
    );


    dim3 blocks(
        N / TILE_SIZE,
        N / TILE_SIZE
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
){

    m.def(
        "tiled_gemm",
        &tiled_gemm,
        "Tiled CUDA GEMM"
    );

}
