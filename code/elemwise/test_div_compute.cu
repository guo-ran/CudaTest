#include "elementwise.cuh"
#include <iostream>

void CudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

template <typename T> struct DivFunctor {
  __device__ T operator()(T x) const {
    return x / static_cast<T>(3.0);
    // return x * static_cast<T>(3.0);
  }
};

template <typename T>
__global__ void FillValue(int64_t elem_cnt, T value, T *ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    ptr[i] = value;
  }
}

int main() {
  using T = float;
  int64_t elem_cnt = 1024 * 1024 * 16;
  T *in_ptr;
  T *out_ptr;
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));
  CudaCheck(cudaMalloc(&in_ptr, elem_cnt * sizeof(T)));
  CudaCheck(cudaMalloc(&out_ptr, elem_cnt * sizeof(T)));
  FillValue<<<elem_cnt / 1024, 1024, 0, stream>>>(elem_cnt, static_cast<T>(4),
                                                  in_ptr);
  CudaCheck(oneflow::cuda::elementwise::Unary<DivFunctor<T>, T, T>(
      DivFunctor<T>(), elem_cnt, out_ptr, in_ptr, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaFree(in_ptr));
  CudaCheck(cudaFree(out_ptr));
  return 0;
}
