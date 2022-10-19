
#include<cuda.h>
#include<cuda_fp16.h>
#include<iostream>
#include <fstream>
#include<vector>
#include "nd_index_helper.h"


void CudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}
const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 8192;

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock, kCudaMaxBlocksNum);
}

#define CUDA_1D_KERNEL_LOOP_T(type, i, n)                                                      \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

template<typename T>
__device__ __forceinline__ bool IsZero(T v) {
  return v == 0;
}

template<>
__device__ __forceinline__ bool IsZero<half>(half v) {
  return v == static_cast<half>(0);
}

template<>
__device__ __forceinline__ bool IsZero<half2>(half2 v) {
  return v.x == static_cast<half>(0) && v.y == static_cast<half>(0);
}

template<typename T, typename K, typename IDX, typename U>
__global__ void UnsortedSegmentRowSumGpu(const IDX data_elem_cnt,
                                         const NdIndexOffsetHelper<IDX, 2> in_helper,
                                         const NdIndexOffsetHelper<IDX, 2> out_helper,
                                         const U* __restrict__ data, const K* __restrict__ segment_ids,
                                         const IDX num_segments, const IDX segment_id_offset,
                                         T* __restrict__ out) {
#pragma unroll
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const U val = data[i];
    if (!IsZero(val)) {
      IDX segment_id_idx, inner_idx;
      in_helper.OffsetToNdIndex(i, segment_id_idx, inner_idx);
      const K origin_idx = segment_ids[segment_id_idx];
      assert(origin_idx >= 0);
      const IDX idx = origin_idx - segment_id_offset;
      if (idx >= 0 && idx < num_segments) {
        const int64_t out_offset = out_helper.NdIndexToOffset(idx, inner_idx);
        if (out_offset >= 0) { 
            atomicAdd(out + out_offset, static_cast<T>(val)); }
      }
    }
  }
}


template<typename T, typename K, typename IDX, typename U>
void UnsortedSegmentSumUtil(cudaStream_t stream, const K* segment_ids, const U* data,
                            IDX num_segment_ids, IDX num_segments, IDX outer_dim_size,
                            IDX inner_dim_size, IDX segment_id_offset, T* out) {
  const IDX data_elem_cnt = num_segment_ids * outer_dim_size * inner_dim_size;

if (outer_dim_size == 1) {
    NdIndexOffsetHelper<IDX, 2> in_helper(num_segment_ids, inner_dim_size);
    NdIndexOffsetHelper<IDX, 2> out_helper(num_segments, inner_dim_size);
    UnsortedSegmentRowSumGpu<T, K, IDX, U>
        <<<BlocksNum4ThreadsNum(data_elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream>>>(data_elem_cnt, in_helper, out_helper,
                                                          data, segment_ids, num_segments,
                                                          segment_id_offset, out);
}
}


template<typename T, typename K, typename IDX, typename U>
void DispatchDataType(cudaStream_t stream, const K* segment_ids, const U* data,
                      int64_t num_segment_ids, int64_t num_segments, int64_t outer_dim_size,
                      int64_t inner_dim_size, int64_t segment_id_offset, T* out) {
  if (std::is_same<T, half>::value && std::is_same<U, half>::value
      && reinterpret_cast<uintptr_t>(data) % sizeof(half2) == 0
      && reinterpret_cast<uintptr_t>(out) % sizeof(half2) == 0 && inner_dim_size % 2 == 0) {
    UnsortedSegmentSumUtil<half2, K, IDX, half2>(
        stream, segment_ids, reinterpret_cast<const half2*>(data), num_segment_ids, num_segments,
        outer_dim_size, inner_dim_size / 2, segment_id_offset, reinterpret_cast<half2*>(out));
  } else {
    //UnsortedSegmentSumUtil<T, K, IDX, U>(stream, segment_ids, data, num_segment_ids, num_segments,
    //                                     outer_dim_size, inner_dim_size, segment_id_offset, out);
  }
}

template<typename T, typename K, typename U>
void UnsortedSegmentSum(cudaStream_t stream, const K* segment_ids, const U* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, T* out) {
    const int64_t data_elem_cnt = num_segment_ids * outer_dim_size * inner_dim_size;
    const int64_t out_elem_cnt = outer_dim_size * num_segments * inner_dim_size;

      DispatchDataType<T, K, int32_t, U>(stream, segment_ids, data, num_segment_ids, num_segments,
                                         outer_dim_size, inner_dim_size, segment_id_offset, out);

  }

int main() {
  int batch_size = 6912;
  int num_ids = batch_size * 26;
  int embedding_size = 128;
  int elem_cnt = num_ids * embedding_size;
  using T = half;

  T *host_grad_ptr;
  T *grad_ptr;
  size_t grad_size = elem_cnt * sizeof(T);
  CudaCheck(cudaMallocHost(&host_grad_ptr, grad_size));
  CudaCheck(cudaMalloc(&grad_ptr, grad_size));
  std::ifstream grad_is;
  grad_is.open("test/grad_0");
  grad_is.read(reinterpret_cast<char *>(host_grad_ptr), grad_size);
  CudaCheck(cudaMemcpy(grad_ptr, host_grad_ptr, grad_size, cudaMemcpyDefault));

  T *host_unique_grad_ptr;
  T *unique_grad_ptr;
  CudaCheck(cudaMallocHost(&host_unique_grad_ptr, grad_size));
  CudaCheck(cudaMalloc(&unique_grad_ptr, grad_size));


  uint32_t *host_indices_ptr;
  uint32_t *indices_ptr;
  size_t indices_size = num_ids * sizeof(uint32_t);
  CudaCheck(cudaMallocHost(&host_indices_ptr, indices_size));
  CudaCheck(cudaMalloc(&indices_ptr, indices_size));
  std::ifstream indices_is;
  indices_is.open("test/indices_0");
  indices_is.read(reinterpret_cast<char *>(host_indices_ptr), indices_size);
  CudaCheck(cudaMemcpy(indices_ptr, host_indices_ptr, indices_size, cudaMemcpyDefault));

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaDeviceSynchronize();
  UnsortedSegmentSum<T, uint32_t, T>(stream, indices_ptr, grad_ptr, num_ids, num_ids, 1, embedding_size, 0, unique_grad_ptr);

  CudaCheck(cudaMemcpy(host_unique_grad_ptr, unique_grad_ptr, grad_size, cudaMemcpyDefault));
  cudaDeviceSynchronize();
  std::ifstream unique_grad_is;
  unique_grad_is.open("test/unique_grad_0");
  std::vector<T> unique_grads_data(elem_cnt);
  unique_grad_is.read(reinterpret_cast<char *>(unique_grads_data.data()), grad_size);
  for (int i = 0; i < 18904*embedding_size; i++) {
    int batch_idx = i / embedding_size;
    int out_i = i % embedding_size;
    float diff = std::abs(static_cast<float>(host_unique_grad_ptr[i]) -
                          static_cast<float>(unique_grads_data.at(i)));
    if (diff > 0.05) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff
                << " out0: " << static_cast<float>(host_unique_grad_ptr[i]) << " out1 "
                << static_cast<float>(unique_grads_data.at(i)) << std::endl;
    }
  }

  return 0;
}
