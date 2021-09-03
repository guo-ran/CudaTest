#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x,                      \
               step = blockDim.x * gridDim.x;                                  \
       i < (n); i += step)
