from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="gemm_cuda_extensions",

    ext_modules=[
        CUDAExtension(
            name="naive_cuda",
            sources=[
               "src/cuda/naive_gemm.cu"
            ],
        ),

        CUDAExtension(
            name="tiled_cuda",
            sources=[
                "src/cuda/tiled_gemm.cu"
            ],
        ),
    ],

    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True
        )
    }
)
