#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <mma.h>
#include <vector>
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

template <typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template <typename T> struct DefaultComputeType { using type = T; };

template <> struct DefaultComputeType<half> { using type = float; };

template <> struct DefaultComputeType<__nv_bfloat16> { using type = float; };

template <typename T, int32_t N> struct DotBwdParam {
  const T *out_grad;
  const T *in[N];
  T *in_grad[N];
  T *output_concat_grad;
  int32_t output_concat_size;
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  int32_t features_dim;
  int32_t num_in;
};


template<typename T, typename AccType, int m, int n, int k, class ALayout, class BLayout>
class Wmma {
 public:
  __device__ void LoadA(const T* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(a_, ptr, ldm); }
  __device__ void LoadB(const T* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(b_, ptr, ldm); }
  __device__ void Store(AccType* ptr, int ldm) {
    nvcuda::wmma::store_matrix_sync(ptr, acc_, ldm, nvcuda::wmma::mem_row_major);
  }
  __device__ void Mma() { nvcuda::wmma::mma_sync(acc_, a_, b_, acc_); }
  __device__ void InitAcc() { nvcuda::wmma::fill_fragment(acc_, 0.0f); }
  __device__ __forceinline__ T Convert(T src) { return src; }

 private:
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, m, n, k, T, ALayout> a_;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, m, n, k, T, BLayout> b_;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, AccType> acc_;
};

template<typename AccType, int m, int n, int k, class ALayout, class BLayout>
class Wmma<float, AccType, m, n, k, ALayout, BLayout> {
 public:
#if __CUDA_ARCH__ >= 800
  __device__ void LoadA(const float* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(a_, ptr, ldm); }
  __device__ void LoadB(const float* ptr, int ldm) { nvcuda::wmma::load_matrix_sync(b_, ptr, ldm); }
  __device__ void Mma() { nvcuda::wmma::mma_sync(acc_, a_, b_, acc_); }
  __device__ __forceinline__ float Convert(float src) { return nvcuda::wmma::__float_to_tf32(src); }
  __device__ void Store(AccType* ptr, int ldm) {
    nvcuda::wmma::store_matrix_sync(ptr, acc_, ldm, nvcuda::wmma::mem_row_major);
  }
  __device__ void InitAcc() { nvcuda::wmma::fill_fragment(acc_, 0.0f); }
#else
  __device__ void LoadA(const float* ptr, int ldm) { __trap(); }
  __device__ void LoadB(const float* ptr, int ldm) { __trap(); }
  __device__ void Mma() { __trap(); }
  __device__ __forceinline__ float Convert(float src) { return src; }
  __device__ void Store(AccType* ptr, int ldm) { __trap(); }
  __device__ void InitAcc() { __trap(); }
#endif

 private:
#if __CUDA_ARCH__ >= 800
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, m, n, k, nvcuda::wmma::precision::tf32, ALayout>
      a_;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, m, n, k, nvcuda::wmma::precision::tf32, BLayout>
      b_;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, m, n, k, AccType> acc_;
#endif
};

constexpr int kUnrollDim = 2;

template<typename T, typename ComputeType, int32_t max_in, int32_t pack_size, int mn_tile_dim,
         int k_tile_dim>
__global__ void DotFeatureInteractionBackwardWmmaImpl(
    int m_num_tiles, int n_num_tiles, int k_num_tiles, int64_t batch_size, int padded_num_rows,
    int vector_num_pack, int padded_vector_num_pack, int out_num_cols, int in_shared_mem_cols,
    int in_shared_mem_cols_num_pack, int matrix_out_grad_shared_mem_cols, int offset,
    DotBwdParam<T, max_in> param) {
#if __CUDA_ARCH__ >= 700
  Wmma<T, ComputeType, mn_tile_dim, mn_tile_dim, k_tile_dim, nvcuda::wmma::row_major,
       nvcuda::wmma::row_major>
      wmma;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T* in_buf = reinterpret_cast<T*>(shared_buf);
  Pack<T, pack_size>* in_buf_pack = reinterpret_cast<Pack<T, pack_size>*>(shared_buf);
  T* matrix_out_grad_buf = in_buf + padded_num_rows * in_shared_mem_cols;
  ComputeType* in_grad_buf = reinterpret_cast<ComputeType*>(
      matrix_out_grad_buf + padded_num_rows * matrix_out_grad_shared_mem_cols);
  Pack<ComputeType, pack_size>* in_grad_buf_pack =
      reinterpret_cast<Pack<ComputeType, pack_size>*>(in_grad_buf);

  int batch_idx = blockIdx.x;
  const T* batch_out_grad = param.out_grad + batch_idx * out_num_cols;
  const int output_concat_size = param.output_concat_size;
  T* batch_output_concat_grad = (param.output_concat_grad)
                                    ? (param.output_concat_grad + batch_idx * output_concat_size)
                                    : nullptr;
  int features_dim = param.features_dim;
  // 1.split out_grad to concat_out_grad and matrix_out_grad buf
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  for (int i = thread_id; i < output_concat_size; i += blockDim.x * blockDim.y) {
    batch_output_concat_grad[i] = batch_out_grad[i];
  }
  const T* batch_interaction_out_grad = batch_out_grad + output_concat_size;
  for (int matrix_row = threadIdx.y; matrix_row < padded_num_rows; matrix_row += blockDim.y) {
    for (int matrix_col = threadIdx.x; matrix_col < padded_num_rows; matrix_col += blockDim.x) {
      const int64_t i = matrix_row * matrix_out_grad_shared_mem_cols + matrix_col;
      T grad_val = 0;
      if (matrix_row < features_dim && matrix_col < features_dim) {
        if (matrix_col < matrix_row) {
          int32_t out_grad_col = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_out_grad[out_grad_col];
        } else if (matrix_row < matrix_col) {
          // transpose add
          int32_t trans_row_id = matrix_col;
          int32_t trans_col_id = matrix_row;
          int32_t out_grad_col =
              trans_row_id * (offset + trans_row_id - 1 + offset) / 2 + trans_col_id;
          grad_val = batch_interaction_out_grad[out_grad_col];
        } else if ((matrix_row == matrix_col) && (offset == 1)) {
          int32_t out_grad_col = matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_out_grad[out_grad_col] * static_cast<T>(2);
        }
      }
      matrix_out_grad_buf[i] = wmma.Convert(grad_val);
    }
  }

  // 2.load in to in in_buf
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < max_in; ++i) {
      if (i >= param.num_in) { break; }
      const Pack<T, pack_size>* batch_in = reinterpret_cast<const Pack<T, pack_size>*>(param.in[i])
                                           + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * kUnrollDim; j < param.in_feature_dim[i];
           j += blockDim.y * kUnrollDim) {
#pragma unroll
        for (int k = 0; k < kUnrollDim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> pack_in_val = batch_in[in_row * vector_num_pack + col];
#pragma unroll
          for (int t = 0; t < pack_size; ++t) {
            pack_in_val.elem[t] = wmma.Convert(pack_in_val.elem[t]);
          }
          in_buf_pack[buf_row * in_shared_mem_cols_num_pack + col] = pack_in_val;
        }
      }
    }
  }
  Pack<T, pack_size> zero;
#pragma unroll
  for (int k = 0; k < pack_size; ++k) { zero.elem[k] = wmma.Convert(0); }
#pragma unroll
  for (int row = features_dim + threadIdx.y; row < padded_num_rows; row += blockDim.y) {
    for (int col = threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      in_buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  for (int row = threadIdx.y; row < features_dim; row += blockDim.y) {
    for (int col = vector_num_pack + threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      in_buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  __syncthreads();

  for (int blocks_id = warp_id; blocks_id < m_num_tiles * n_num_tiles; blocks_id += blockDim.y) {
    int blocks_row = blocks_id / n_num_tiles;
    int blocks_col = blocks_id - blocks_row * n_num_tiles;
    wmma.InitAcc();
    for (int step = 0; step < k_num_tiles; ++step) {
      // blocks_row is a row_id, step is a col_id. blocks_col is b col_id,
      // step is b row_id.
      T* tile_a_ptr = matrix_out_grad_buf
                      + blocks_row * mn_tile_dim * matrix_out_grad_shared_mem_cols
                      + step * k_tile_dim;
      T* tile_b_ptr = in_buf + step * k_tile_dim * in_shared_mem_cols + blocks_col * mn_tile_dim;
      wmma.LoadA(tile_a_ptr, matrix_out_grad_shared_mem_cols);
      wmma.LoadB(tile_b_ptr, in_shared_mem_cols);
      wmma.Mma();
    }
    ComputeType* tile_ptr =
        in_grad_buf + blocks_row * mn_tile_dim * in_shared_mem_cols + blocks_col * mn_tile_dim;
    wmma.Store(tile_ptr, in_shared_mem_cols);
  }
  __syncthreads();

  // 4.split in_grad buf to dx
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < max_in; ++i) {
      if (i >= param.num_in) { break; }
      Pack<T, pack_size>* batch_in_grad = reinterpret_cast<Pack<T, pack_size>*>(param.in_grad[i])
                                          + batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * kUnrollDim; j < param.in_feature_dim[i];
           j += blockDim.y * kUnrollDim) {
#pragma unroll
        for (int k = 0; k < kUnrollDim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) { break; }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> grad_val;
          Pack<ComputeType, pack_size> buf_grad_val =
              in_grad_buf_pack[buf_row * in_shared_mem_cols_num_pack + col];
#pragma unroll
          for (int t = 0; t < pack_size; ++t) {
            grad_val.elem[t] = static_cast<T>(buf_grad_val.elem[t]);
          }
          batch_in_grad[in_row * vector_num_pack + col] = grad_val;
        }
      }
    }
  }
#else
  __trap();
#endif  // __CUDA_ARCH__ >= 700
}


template <typename T> struct KTileDim { static const int val = 16; };

template <> struct KTileDim<float> { static const int val = 8; };

int main() {
  using T = half; // if T is float, sm version must > 800
  int64_t batch_size = 55296 / 8;
  int64_t vector_size = 128;
  int64_t out_num_cols = 480;
  std::vector<int64_t> feature_dims = {1, 26};
  const int pack_size = 4;
  const int features_dim = 27;
  const int concated_padded_dim = 32;
  const int TILE_DIM = 16;
  const int K_TILE_DIM = KTileDim<T>::val;
  const int m_num_tiles = concated_padded_dim / TILE_DIM;
  const int n_num_tiles = vector_size / TILE_DIM;
  const int k_num_tiles = concated_padded_dim / K_TILE_DIM;
  const int skew_half = 8;
  const int skew_acc = 8;
  const int block_size = 256;
  const int block_dim_x = 32;
  const int block_dim_y = block_size / block_dim_x;
  const int num_num_tiles = batch_size;
  T *host_in_0_ptr;
  T *in_0_ptr;
  size_t in_0_size = batch_size * feature_dims.at(0) * vector_size * sizeof(T);
  CudaCheck(cudaMallocHost(&host_in_0_ptr, in_0_size));
  CudaCheck(cudaMalloc(&in_0_ptr, in_0_size));
  T *host_output_concat_grad_ptr;
  T *output_concat_grad_ptr;
  CudaCheck(cudaMallocHost(&host_output_concat_grad_ptr, in_0_size));
  CudaCheck(cudaMalloc(&output_concat_grad_ptr, in_0_size));
  T *host_in_1_ptr;
  T *in_1_ptr;
  size_t in_1_size = batch_size * feature_dims.at(1) * vector_size * sizeof(T);
  CudaCheck(cudaMallocHost(&host_in_1_ptr, in_1_size));
  CudaCheck(cudaMalloc(&in_1_ptr, in_1_size));
  int32_t* host_sparse_ids_ptr;
  int32_t* sparse_ids_ptr;
  size_t sparse_ids_size = batch_size * feature_dims.at(1) * sizeof(int32_t);
  CudaCheck(cudaMallocHost(&host_sparse_ids_ptr, sparse_ids_size));
  CudaCheck(cudaMalloc(&sparse_ids_ptr, sparse_ids_size));
  T *host_dy_ptr;
  T *dy_ptr;
  int64_t out_dim = 480;
  size_t out_size = batch_size * out_dim * sizeof(T);
  CudaCheck(cudaMalloc(&dy_ptr, out_size));
  CudaCheck(cudaMallocHost(&host_dy_ptr, out_size));
  T *host_in_0_grad_ptr;
  T *in_0_grad_ptr;
  CudaCheck(cudaMallocHost(&host_in_0_grad_ptr, in_0_size));
  CudaCheck(cudaMalloc(&in_0_grad_ptr, in_0_size));
  T *host_in_1_grad_ptr;
  T *in_1_grad_ptr;
  CudaCheck(cudaMallocHost(&host_in_1_grad_ptr, in_1_size));
  CudaCheck(cudaMalloc(&in_1_grad_ptr, in_1_size));
  T *host_sparse_in_grad_ptr;
  T *sparse_in_grad_ptr;
  size_t sparse_in_size = 4565248 * sizeof(T);
  CudaCheck(cudaMallocHost(&host_sparse_in_grad_ptr, sparse_in_size));
  CudaCheck(cudaMalloc(&sparse_in_grad_ptr, sparse_in_size));

  DotBwdParam<T, 2> param;
  param.in[0] = in_0_ptr;
  param.in[1] = in_1_ptr;
  param.in_grad[0] = in_0_grad_ptr;
  param.in_grad[1] = in_1_grad_ptr;
  param.in_feature_dim[0] = feature_dims.at(0);
  param.in_feature_dim[1] = feature_dims.at(1);
  param.dim_start_offset[0] = 0;
  param.dim_start_offset[1] = feature_dims.at(0);
  param.num_in = 2;
  param.out_grad = dy_ptr;
  param.features_dim = features_dim;
  param.output_concat_grad = output_concat_grad_ptr;
  param.output_concat_size = vector_size;
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  std::ifstream in_0_is;
  in_0_is.open("in_0.bin");
  in_0_is.read(reinterpret_cast<char *>(host_in_0_ptr), in_0_size);
  CudaCheck(cudaMemcpy(in_0_ptr, host_in_0_ptr, in_0_size, cudaMemcpyDefault));

  std::ifstream in_1_is;
  in_1_is.open("in_1.bin");
  in_1_is.read(reinterpret_cast<char *>(host_in_1_ptr), in_1_size);
  CudaCheck(cudaMemcpy(in_1_ptr, host_in_1_ptr, in_1_size, cudaMemcpyDefault));

    std::ifstream sparse_ids_is;
  sparse_ids_is.open("sparse_ids.bin");
  sparse_ids_is.read(reinterpret_cast<char *>(host_sparse_ids_ptr), sparse_ids_size);
  CudaCheck(cudaMemcpy(sparse_ids_ptr, host_sparse_ids_ptr, sparse_ids_size, cudaMemcpyDefault));

  std::ifstream dy_is;
  dy_is.open("dy.bin");
  dy_is.read(reinterpret_cast<char *>(host_dy_ptr), out_size);
  CudaCheck(cudaMemcpy(dy_ptr, host_dy_ptr, out_size, cudaMemcpyDefault));

  const int in_shared_mem_num_cols = vector_size + skew_half;
  const int matrix_dy_shared_mem_cols = concated_padded_dim + skew_acc;
  const size_t in_shared_mem_bytes =
      concated_padded_dim * in_shared_mem_num_cols * sizeof(T);
  const size_t matrix_dy_shared_mem_bytes =
      concated_padded_dim * matrix_dy_shared_mem_cols * sizeof(T);
  using ComputeType = typename DefaultComputeType<T>::type;
  const size_t in_grad_shared_mem_bytes =
      concated_padded_dim * in_shared_mem_num_cols * sizeof(ComputeType);
  const size_t warp_shared_mem_bytes = in_shared_mem_bytes +
                                       matrix_dy_shared_mem_bytes +
                                       in_grad_shared_mem_bytes;

  const int vector_num_pack = vector_size / pack_size;
  const int in_shared_mem_cols_num_pack = in_shared_mem_num_cols / pack_size;

  DotFeatureInteractionBackwardWmmaImpl<T, ComputeType, 2, 4, TILE_DIM,
                                          K_TILE_DIM>
      <<<num_num_tiles, dim3(block_dim_x, block_dim_y), warp_shared_mem_bytes,
         stream>>>(m_num_tiles, n_num_tiles, k_num_tiles, batch_size,
                   concated_padded_dim, vector_num_pack, vector_num_pack,
                   out_num_cols, in_shared_mem_num_cols,
                   in_shared_mem_cols_num_pack, matrix_dy_shared_mem_cols, 0,
                   param);
  int64_t num_ids = 6912 * 26;
  UnsortedSegmentSum<T, int32_t, T>(stream, sparse_ids_ptr, in_1_grad_ptr, num_ids, num_ids, 1, 128, 0, sparse_in_grad_ptr);
  

  CudaCheck(cudaMemcpy(host_in_0_grad_ptr, in_0_grad_ptr, in_0_size,
                       cudaMemcpyDefault));
  CudaCheck(cudaMemcpy(host_in_1_grad_ptr, in_1_grad_ptr, in_1_size,
                       cudaMemcpyDefault));
  CudaCheck(cudaMemcpy(host_output_concat_grad_ptr, output_concat_grad_ptr,
                       in_0_size, cudaMemcpyDefault));

  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaDeviceSynchronize());
  //std::ifstream in_0_grad_is;
  //in_0_grad_is.open("in_0_grad.bin");
  //std::vector<T> in_0_grad_data(batch_size * feature_dims.at(0) * vector_size);
  //in_0_grad_is.read(reinterpret_cast<char *>(in_0_grad_data.data()), in_0_size);
//
  //for (int i = 0; i < batch_size * feature_dims.at(0) * vector_size; i++) {
  //  int batch_idx = i / (feature_dims.at(0) * vector_size);
  //  int out_i = i % (feature_dims.at(0) * vector_size);
  //  float diff = std::abs(static_cast<float>(host_in_0_grad_ptr[i]) -
  //                        static_cast<float>(in_0_grad_data.at(i)));
  //  if (diff > 0.001) {
  //    std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
  //              << " diff " << diff
  //              << " out0: " << static_cast<float>(host_in_0_grad_ptr[i])
  //              << " out1 " << static_cast<float>(in_0_grad_data.at(i))
  //              << std::endl;
  //  }
  //}
//
  //std::ifstream in_1_grad_is;
  //in_1_grad_is.open("in_1_grad.bin");
  //std::vector<T> in_1_grad_data(batch_size * feature_dims.at(1) * vector_size);
  //in_1_grad_is.read(reinterpret_cast<char *>(in_1_grad_data.data()), in_1_size);
//
  //for (int i = 0; i < batch_size * feature_dims.at(1) * vector_size; i++) {
  //  int batch_idx = i / (feature_dims.at(1) * vector_size);
  //  int out_i = i % (feature_dims.at(1) * vector_size);
  //  float diff = std::abs(static_cast<float>(host_in_1_grad_ptr[i]) -
  //                        static_cast<float>(in_1_grad_data.at(i)));
  //  if (diff > 0.001) {
  //    std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
  //              << " diff " << diff
  //              << " out0: " << static_cast<float>(host_in_1_grad_ptr[i])
  //              << " out1 " << static_cast<float>(in_1_grad_data.at(i))
  //              << std::endl;
  //  }
  //}
//
  //std::ifstream output_concat_grad_is;
  //output_concat_grad_is.open("output_concat_grad.bin");
  //std::vector<T> output_concat_grad_data(batch_size * vector_size);
  //output_concat_grad_is.read(
  //    reinterpret_cast<char *>(output_concat_grad_data.data()), in_0_size);
//
  //for (int i = 0; i < batch_size * vector_size; i++) {
  //  int batch_idx = i / (vector_size);
  //  int out_i = i % (vector_size);
  //  float diff = std::abs(static_cast<float>(host_output_concat_grad_ptr[i]) -
  //                        static_cast<float>(output_concat_grad_data.at(i)));
  //  if (diff > 0.001) {
  //    std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
  //              << " diff " << diff << " out0: "
  //              << static_cast<float>(host_output_concat_grad_ptr[i])
  //              << " out1 " << static_cast<float>(output_concat_grad_data.at(i))
  //              << std::endl;
  //  }
  //}
  CudaCheck(cudaFree(in_0_ptr));
  CudaCheck(cudaFreeHost(host_in_0_ptr));
  CudaCheck(cudaFreeHost(host_in_1_ptr));
  CudaCheck(cudaFree(in_1_ptr));
  CudaCheck(cudaFree(dy_ptr));
  CudaCheck(cudaFreeHost(host_dy_ptr));
  return 0;
}
