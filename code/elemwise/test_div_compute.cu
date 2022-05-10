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

template <typename T, int pack_size> struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T),
                                             pack_size * sizeof(T)>::type;
};

template <typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template <typename T, int pack_size> union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template <typename T>
__global__ void PackCopyKernel(int64_t elem_cnt, T value, const T *in_ptr,
                               T *out_ptr) {
  const Pack<T, 4> *src_pack = reinterpret_cast<const Pack<T, 4> *>(in_ptr);
  Pack<T, 4> *dst_pack = reinterpret_cast<Pack<T, 4> *>(out_ptr);
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    Pack<T, 4> src = src_pack[i];
    Pack<T, 4> dst;
    for (int j = 0; j < 4; j++) {
      dst.elem[j] = src.elem[j];
    }
    dst_pack[i] = dst;
  }
}

template <typename T>
__global__ void PackDivKernel(int64_t elem_cnt, T value, const T *in_ptr,
                              T *out_ptr) {
  const Pack<T, 4> *src_pack = reinterpret_cast<const Pack<T, 4> *>(in_ptr);
  Pack<T, 4> *dst_pack = reinterpret_cast<Pack<T, 4> *>(out_ptr);
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    Pack<T, 4> src = src_pack[i];
    Pack<T, 4> dst;
    for (int j = 0; j < 4; j++) {
      dst.elem[j] = src.elem[j] / value;
    }
    dst_pack[i] = dst;
  }
}

template <typename T>
__global__ void NotPackCopyKernel(int64_t elem_cnt, T value, const T *in_ptr,
                                  T *out_ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    out_ptr[i] = in_ptr[i];
  }
}

template <typename T>
__global__ void NotPackDivKernel(int64_t elem_cnt, T value, const T *in_ptr,
                                 T *out_ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    out_ptr[i] = in_ptr[i] / value;
  }
}

template <typename T>
__global__ void FillValue(int64_t elem_cnt, T value, T *ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    ptr[i] = value;
  }
}

int main() {
  using T = float; // int
  int64_t elem_cnt = 1024 * 1024 * 16;
  T *in_ptr;
  T *out_ptr;
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));
  CudaCheck(cudaMalloc(&in_ptr, elem_cnt * sizeof(T)));
  CudaCheck(cudaMalloc(&out_ptr, elem_cnt * sizeof(T)));
  FillValue<<<elem_cnt / 1024, 1024, 0, stream>>>(elem_cnt, static_cast<T>(4),
                                                  in_ptr);
  NotPackCopyKernel<<<elem_cnt / 1024, 1024, 0, stream>>>(
      elem_cnt, static_cast<T>(2), in_ptr, out_ptr);
  NotPackDivKernel<<<elem_cnt / 1024, 1024, 0, stream>>>(
      elem_cnt, static_cast<T>(2), in_ptr, out_ptr);
  PackCopyKernel<<<elem_cnt / 4 / 1024, 1024, 0, stream>>>(
      elem_cnt / 4, static_cast<T>(2), in_ptr, out_ptr);
  PackDivKernel<<<elem_cnt / 4 / 1024, 1024, 0, stream>>>(
      elem_cnt / 4, static_cast<T>(2), in_ptr, out_ptr);
  CudaCheck(oneflow::cuda::elementwise::Unary<DivFunctor<T>, T, T>(
      DivFunctor<T>(), elem_cnt, out_ptr, in_ptr, stream));
  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaFree(in_ptr));
  CudaCheck(cudaFree(out_ptr));
  return 0;
}
