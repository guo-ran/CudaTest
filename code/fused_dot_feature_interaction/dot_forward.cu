#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <mma.h>
#include <vector>

void CudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

template <typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template <typename T> struct DefaultComputeType { using type = T; };

template <> struct DefaultComputeType<half> { using type = float; };

template <> struct DefaultComputeType<__nv_bfloat16> { using type = float; };

template <typename T, int32_t N> struct DotFwdParam {
  const T *in[N];
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  int32_t features_dim;
  const T *output_concat;
  int32_t output_concat_size;
  T *out;
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
__global__ void DotFeatureInteractionWmmaImpl(
    int m_num_tiles, int k_num_tiles, int64_t batch_size, int padded_num_rows, int vector_num_pack,
    int padded_vector_num_pack, int out_num_cols, int out_num_cols_num_pack, int in_shared_mem_cols,
    int in_shared_mem_cols_num_pack, int acc_shared_mem_cols, int acc_shared_mem_cols_num_pack,
    int offset, int output_padding, DotFwdParam<T, max_in> param) {
#if __CUDA_ARCH__ >= 700
  Wmma<T, ComputeType, mn_tile_dim, mn_tile_dim, k_tile_dim, nvcuda::wmma::row_major,
       nvcuda::wmma::col_major>
      wmma;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T* buf = reinterpret_cast<T*>(shared_buf);
  Pack<T, pack_size>* buf_pack = reinterpret_cast<Pack<T, pack_size>*>(shared_buf);
  ComputeType* acc_buf =
      reinterpret_cast<ComputeType*>(shared_buf + padded_num_rows * in_shared_mem_cols * sizeof(T));
  int batch_idx = blockIdx.x;
  T* batch_out = param.out + batch_idx * out_num_cols;
  Pack<T, pack_size>* batch_out_pack =
      reinterpret_cast<Pack<T, pack_size>*>(param.out) + batch_idx * out_num_cols_num_pack;
  const int output_concat_size = param.output_concat_size;
  const T* batch_output_concat =
      (param.output_concat) ? (param.output_concat + batch_idx * output_concat_size) : nullptr;
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
          buf_pack[buf_row * in_shared_mem_cols_num_pack + col] = pack_in_val;
        }
      }
    }
  }
  Pack<T, pack_size> zero;
#pragma unroll
  for (int k = 0; k < pack_size; ++k) { zero.elem[k] = wmma.Convert(0); }
  for (int row = threadIdx.y; row < param.features_dim; row += blockDim.y) {
    for (int col = vector_num_pack + threadIdx.x; col < padded_vector_num_pack; col += blockDim.x) {
      buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  __syncthreads();
  for (int blocks_id = warp_id; blocks_id < m_num_tiles * m_num_tiles; blocks_id += blockDim.y) {
    int blocks_row = blocks_id / m_num_tiles;
    int blocks_col = blocks_id - blocks_row * m_num_tiles;
    if (blocks_row >= blocks_col) {
      wmma.InitAcc();
      for (int step = 0; step < k_num_tiles; ++step) {
        T* tile_a_ptr = buf + blocks_row * mn_tile_dim * in_shared_mem_cols + step * k_tile_dim;
        T* tile_b_ptr = buf + blocks_col * mn_tile_dim * in_shared_mem_cols + step * k_tile_dim;
        wmma.LoadA(tile_a_ptr, in_shared_mem_cols);
        wmma.LoadB(tile_b_ptr, in_shared_mem_cols);
        wmma.Mma();
      }
      ComputeType* tile_ptr =
          acc_buf + blocks_row * mn_tile_dim * acc_shared_mem_cols + blocks_col * mn_tile_dim;
      wmma.Store(tile_ptr, acc_shared_mem_cols);
    }
  }
  __syncthreads();
  T* emb_out = batch_out + output_concat_size;
  for (int base_row = threadIdx.y * kUnrollDim; base_row < param.features_dim;
       base_row += kUnrollDim * blockDim.y) {
#pragma unroll
    for (int k = 0; k < kUnrollDim; ++k) {
      int row = base_row + k;
      if (row >= param.features_dim) { break; }
      for (int col = threadIdx.x; col < param.features_dim; col += blockDim.x) {
        if (col < row + offset) {
          int64_t idx = row * (offset + row - 1 + offset) / 2 + col;
          emb_out[idx] = static_cast<T>(acc_buf[row * acc_shared_mem_cols + col]);
        }
      }
    }
  }
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < output_concat_size; i += blockDim.x * blockDim.y) {
    batch_out[i] = batch_output_concat[i];
  }
  for (int i = thread_id; i < output_padding; i += blockDim.x * blockDim.y) {
    batch_out[out_num_cols - 1 - i] = 0;
  }
#else
  __trap();
#endif  // __CUDA_ARCH__ >= 700
}

template <typename T> struct KTileDim { static const int val = 16; };

template <> struct KTileDim<float> { static const int val = 8; };

int main() {
  using T = half; // can be float/half/nv_bfloat16, if T is float, sm version
                   // must > 800
  using ComputeType = typename DefaultComputeType<T>::type;
  int64_t batch_size = 55296 / 8;
  int64_t vector_size = 128;
  int64_t out_num_cols = 480;
  std::vector<int64_t> feature_dims = {1, 26};
  const int features_dim = feature_dims.at(0) + feature_dims.at(1);
  const int concated_padded_dim = 32;
  const int block_size = 128;
  const int pack_size = 4;
  const int TILE_DIM = 16;
  const int K_TILE_DIM = KTileDim<T>::val;
  const int m_num_tiles = concated_padded_dim / TILE_DIM;
  const int k_num_tiles = vector_size / K_TILE_DIM;
  const int skew_half = 8;
  const int skew_acc = 8;

  const int block_dim_x = 32;
  const int block_dim_y = block_size / block_dim_x;
  const int num_num_tiles = batch_size;

  T *host_in_0_ptr;
  T *in_0_ptr;
  size_t in_0_size = batch_size * feature_dims.at(0) * vector_size * sizeof(T);
  CudaCheck(cudaMallocHost(&host_in_0_ptr, in_0_size));
  CudaCheck(cudaMalloc(&in_0_ptr, in_0_size));
  T *host_in_1_ptr;
  T *in_1_ptr;
  size_t in_1_size = batch_size * feature_dims.at(1) * vector_size * sizeof(T);
  CudaCheck(cudaMallocHost(&host_in_1_ptr, in_1_size));
  CudaCheck(cudaMalloc(&in_1_ptr, in_1_size));
  T *host_output_concat_ptr;
  T *output_concat_ptr;
  CudaCheck(cudaMallocHost(&host_output_concat_ptr, in_0_size));
  CudaCheck(cudaMalloc(&output_concat_ptr, in_0_size));
  T *host_out_ptr;
  T *out_ptr;
  size_t out_size = batch_size * out_num_cols * sizeof(T);
  CudaCheck(cudaMalloc(&out_ptr, out_size));
  CudaCheck(cudaMallocHost(&host_out_ptr, out_size));

  DotFwdParam<T, 2> param;
  param.in[0] = in_0_ptr;
  param.in[1] = in_1_ptr;
  param.in_feature_dim[0] = feature_dims.at(0);
  param.in_feature_dim[1] = feature_dims.at(1);
  param.dim_start_offset[0] = 0;
  param.dim_start_offset[1] = feature_dims.at(0);
  param.num_in = 2;
  param.out = out_ptr;
  param.features_dim = features_dim;
  param.output_concat = output_concat_ptr;
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

  std::ifstream output_concat_is;
  output_concat_is.open("output_concat.bin");
  output_concat_is.read(reinterpret_cast<char *>(host_output_concat_ptr),
                        in_0_size);
  CudaCheck(cudaMemcpy(output_concat_ptr, host_output_concat_ptr, in_0_size,
                       cudaMemcpyDefault));

  const int in_shared_mem_num_cols = vector_size + skew_half;
  const int acc_shared_mem_num_cols = concated_padded_dim + skew_acc;
  const size_t in_shared_mem_bytes =
      concated_padded_dim * in_shared_mem_num_cols * sizeof(T);
  const size_t acc_shared_mem_bytes =
      concated_padded_dim * acc_shared_mem_num_cols * sizeof(ComputeType);
  const size_t warp_shared_mem_bytes =
      in_shared_mem_bytes + acc_shared_mem_bytes;

  const int out_num_cols_num_pack = out_num_cols / pack_size;
  const int vector_num_pack = vector_size / pack_size;
  const int in_shared_mem_cols_num_pack = in_shared_mem_num_cols / pack_size;
  const int acc_shared_mem_cols_num_pack = acc_shared_mem_num_cols / pack_size;

  DotFeatureInteractionWmmaImpl<T, ComputeType, 2, 4, TILE_DIM, K_TILE_DIM>
      <<<num_num_tiles, dim3(block_dim_x, block_dim_y), warp_shared_mem_bytes,
         stream>>>(m_num_tiles, k_num_tiles, batch_size, concated_padded_dim,
                   vector_num_pack, vector_num_pack, out_num_cols,
                   out_num_cols_num_pack, in_shared_mem_num_cols,
                   in_shared_mem_cols_num_pack, acc_shared_mem_num_cols,
                   acc_shared_mem_cols_num_pack, 0, 1, param);
  CudaCheck(cudaMemcpy(host_out_ptr, out_ptr, out_size, cudaMemcpyDefault));

  std::ifstream out_is;
  out_is.open("out.bin");
  std::vector<T> out_data(batch_size * out_num_cols);
  out_is.read(reinterpret_cast<char *>(out_data.data()), out_size);

  for (int i = 0; i < batch_size * out_num_cols; i++) {
    int batch_idx = i / out_num_cols;
    int out_i = i % out_num_cols;
    float diff = std::abs(static_cast<float>(host_out_ptr[i]) -
                          static_cast<float>(out_data.at(i)));
    if (diff > 0.001) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff
                << " out0: " << static_cast<float>(host_out_ptr[i]) << " out1 "
                << static_cast<float>(out_data.at(i)) << std::endl;
    }
  }

  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaDeviceSynchronize());
  CudaCheck(cudaFree(in_0_ptr));
  CudaCheck(cudaFreeHost(host_in_0_ptr));
  CudaCheck(cudaFreeHost(host_in_1_ptr));
  CudaCheck(cudaFree(in_1_ptr));
  CudaCheck(cudaFree(out_ptr));
  CudaCheck(cudaFreeHost(host_out_ptr));
  return 0;
}
