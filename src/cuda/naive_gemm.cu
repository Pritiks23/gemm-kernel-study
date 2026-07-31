#include <torch/extension.h>


__global__ void naive_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if(row < N && col < N){

        float value = 0.0f;


        for(int k = 0; k < N; k++){

            value += A[row * N + k] *
                     B[k * N + col];

        }


        C[row * N + col] = value;
    }
}



torch::Tensor naive_gemm(
    torch::Tensor A,
    torch::Tensor B
){

    int N = A.size(0);


    auto C = torch::zeros(
        {N, N},
        A.options()
    );


    dim3 threads(
        16,
        16
    );


    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y
    );


    naive_gemm_kernel<<<blocks, threads>>>(
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
        "naive_gemm",
        &naive_gemm,
        "Naive CUDA GEMM"
    );

}
