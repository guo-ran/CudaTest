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

template <typename T, int32_t N> struct DotBwdParam {
  const T *dy;
  const T *in[N];
  T *in_grad[N];
  T *output_concat_grad;
  int32_t output_concat_size;
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  int32_t features_dim;
  int32_t num_in;
};

template <typename T> struct DefaultPrecision { using type = T; };

#if __CUDA_ARCH__ >= 800
template <> struct DefaultPrecision<float> {
  using type = nvcuda::wmma::precision::tf32;
};
#endif

template <typename T> __device__ T ConvertToPrecision(T src) { return src; }

#if __CUDA_ARCH__ >= 800
template <> __device__ float ConvertToPrecision<float>(float src) {
  return nvcuda::wmma::__float_to_tf32(src);
}
#endif

constexpr int unroll_dim = 2;

template <typename T, typename ComputeType, int32_t N, int32_t pack_size,
          int tile_dim, int k_tile_dim>
__global__ void DotFeatureInteractionBackwardTensorCore(
    int m_num_tiles, int n_num_tiles, int k_num_tiles, int64_t batch_size,
    int padded_num_rows, int vector_num_pack, int padded_vector_num_pack,
    int out_num_cols, int in_shared_mem_cols, int in_shared_mem_cols_num_pack,
    int matrix_dy_shared_mem_cols, int offset, DotBwdParam<T, N> param) {
  using PrecisionType = typename DefaultPrecision<T>::type;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T *in_buf = reinterpret_cast<T *>(shared_buf);
  Pack<T, pack_size> *in_buf_pack =
      reinterpret_cast<Pack<T, pack_size> *>(shared_buf);
  T *matrix_dy_buf = in_buf + padded_num_rows * in_shared_mem_cols;
  ComputeType *in_grad_buf = reinterpret_cast<ComputeType *>(
      matrix_dy_buf + padded_num_rows * matrix_dy_shared_mem_cols);
  Pack<ComputeType, pack_size> *in_grad_buf_pack =
      reinterpret_cast<Pack<ComputeType, pack_size> *>(in_grad_buf);

  int batch_idx = blockIdx.x;
  const T *batch_dy = param.dy + batch_idx * out_num_cols;
  const int output_concat_size = param.output_concat_size;
  T *batch_output_concat_grad =
      (param.output_concat_grad)
          ? (param.output_concat_grad + batch_idx * output_concat_size)
          : nullptr;
  int features_dim = param.features_dim;
  // 1.split dy to concat_out_grad and matrix_dy buf
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  for (int i = thread_id; i < output_concat_size;
       i += blockDim.x * blockDim.y) {
    batch_output_concat_grad[i] = batch_dy[i];
  }
  const T *batch_interaction_dy = batch_dy + output_concat_size;
  for (int matrix_row = threadIdx.y; matrix_row < padded_num_rows;
       matrix_row += blockDim.y) {
    for (int matrix_col = threadIdx.x; matrix_col < padded_num_rows;
         matrix_col += blockDim.x) {
      const int64_t i = matrix_row * matrix_dy_shared_mem_cols + matrix_col;
      T grad_val = 0;
      if (matrix_row < features_dim && matrix_col < features_dim) {
        if (matrix_col < matrix_row) {
          int32_t dy_col_idx =
              matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_dy[dy_col_idx];
        } else if (matrix_row < matrix_col) {
          // transpose add
          int32_t trans_row_id = matrix_col;
          int32_t trans_col_id = matrix_row;
          int32_t dy_col_idx =
              trans_row_id * (offset + trans_row_id - 1 + offset) / 2 +
              trans_col_id;
          grad_val = batch_interaction_dy[dy_col_idx];
        } else if ((matrix_row == matrix_col) && (offset == 1)) {
          int32_t dy_col_idx =
              matrix_row * (offset + matrix_row - 1 + offset) / 2 + matrix_col;
          grad_val = batch_interaction_dy[dy_col_idx] * static_cast<T>(2);
        }
      }
      matrix_dy_buf[i] = ConvertToPrecision<T>(grad_val);
    }
  }

  // 2.load in to in in_buf
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= param.num_in) {
        break;
      }
      const Pack<T, pack_size> *batch_in =
          reinterpret_cast<const Pack<T, pack_size> *>(param.in[i]) +
          batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * unroll_dim; j < param.in_feature_dim[i];
           j += blockDim.y * unroll_dim) {
#pragma unroll
        for (int k = 0; k < unroll_dim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) {
            break;
          }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> pack_in_val =
              batch_in[in_row * vector_num_pack + col];
          for (int t = 0; t < pack_size; ++t) {
            pack_in_val.elem[t] = ConvertToPrecision<T>(pack_in_val.elem[t]);
          }
          in_buf_pack[buf_row * in_shared_mem_cols_num_pack + col] =
              pack_in_val;
        }
      }
    }
  }
  Pack<T, pack_size> zero;
  for (int k = 0; k < pack_size; ++k) {
    zero.elem[k] = ConvertToPrecision<T>(0);
  }
#pragma unroll
  for (int row = features_dim + threadIdx.y; row < padded_num_rows;
       row += blockDim.y) {
    for (int col = threadIdx.x; col < padded_vector_num_pack;
         col += blockDim.x) {
      in_buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  for (int row = threadIdx.y; row < features_dim; row += blockDim.y) {
    for (int col = vector_num_pack + threadIdx.x; col < padded_vector_num_pack;
         col += blockDim.x) {
      in_buf_pack[row * in_shared_mem_cols_num_pack + col] = zero;
    }
  }
  __syncthreads();

  for (int blocks_id = warp_id; blocks_id < m_num_tiles * n_num_tiles;
       blocks_id += blockDim.y) {
    int blocks_row_id = blocks_id / n_num_tiles;
    int blocks_col_id = blocks_id - blocks_row_id * n_num_tiles;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tile_dim, tile_dim,
                           k_tile_dim, ComputeType>
        acc;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tile_dim, tile_dim,
                           k_tile_dim, PrecisionType, nvcuda::wmma::row_major>
        a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tile_dim, tile_dim,
                           k_tile_dim, PrecisionType, nvcuda::wmma::row_major>
        b;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int step = 0; step < k_num_tiles; ++step) {
      // blocks_row_id is a row_id, step is a col_id. blocks_col_id is b col_id,
      // step is b row_id.
      T *tile_a_ptr = matrix_dy_buf +
                      blocks_row_id * tile_dim * matrix_dy_shared_mem_cols +
                      step * k_tile_dim;
      T *tile_b_ptr = in_buf + step * k_tile_dim * in_shared_mem_cols +
                      blocks_col_id * tile_dim;
      nvcuda::wmma::load_matrix_sync(a, tile_a_ptr, matrix_dy_shared_mem_cols);
      nvcuda::wmma::load_matrix_sync(b, tile_b_ptr, in_shared_mem_cols);
      nvcuda::wmma::mma_sync(acc, a, b, acc);
    }
    ComputeType *tile_ptr = in_grad_buf +
                            blocks_row_id * tile_dim * in_shared_mem_cols +
                            blocks_col_id * tile_dim;
    nvcuda::wmma::store_matrix_sync(tile_ptr, acc, in_shared_mem_cols,
                                    nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  // 4.split in_grad buf to dx
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= param.num_in) {
        break;
      }
      Pack<T, pack_size> *batch_in_grad =
          reinterpret_cast<Pack<T, pack_size> *>(param.in_grad[i]) +
          batch_idx * param.in_feature_dim[i] * vector_num_pack;
      for (int j = threadIdx.y * unroll_dim; j < param.in_feature_dim[i];
           j += blockDim.y * unroll_dim) {
#pragma unroll
        for (int k = 0; k < unroll_dim; ++k) {
          int in_row = j + k;
          if (in_row >= param.in_feature_dim[i]) {
            break;
          }
          int buf_row = param.dim_start_offset[i] + in_row;
          Pack<T, pack_size> grad_val;
          Pack<ComputeType, pack_size> buf_grad_val =
              in_grad_buf_pack[buf_row * in_shared_mem_cols_num_pack + col];
          for (int t = 0; t < pack_size; ++t) {
            grad_val.elem[t] = static_cast<T>(buf_grad_val.elem[t]);
          }
          batch_in_grad[in_row * vector_num_pack + col] = grad_val;
        }
      }
    }
  }
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
  param.dy = dy_ptr;
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

  DotFeatureInteractionBackwardTensorCore<T, ComputeType, 2, 4, TILE_DIM,
                                          K_TILE_DIM>
      <<<num_num_tiles, dim3(block_dim_x, block_dim_y), warp_shared_mem_bytes,
         stream>>>(m_num_tiles, n_num_tiles, k_num_tiles, batch_size,
                   concated_padded_dim, vector_num_pack, vector_num_pack,
                   out_num_cols, in_shared_mem_num_cols,
                   in_shared_mem_cols_num_pack, matrix_dy_shared_mem_cols, 0,
                   param);

  CudaCheck(cudaMemcpy(host_in_0_grad_ptr, in_0_grad_ptr, in_0_size,
                       cudaMemcpyDefault));
  CudaCheck(cudaMemcpy(host_in_1_grad_ptr, in_1_grad_ptr, in_1_size,
                       cudaMemcpyDefault));
  CudaCheck(cudaMemcpy(host_output_concat_grad_ptr, output_concat_grad_ptr,
                       in_0_size, cudaMemcpyDefault));

  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaDeviceSynchronize());
  std::ifstream in_0_grad_is;
  in_0_grad_is.open("in_0_grad.bin");
  std::vector<T> in_0_grad_data(batch_size * feature_dims.at(0) * vector_size);
  in_0_grad_is.read(reinterpret_cast<char *>(in_0_grad_data.data()), in_0_size);

  for (int i = 0; i < batch_size * feature_dims.at(0) * vector_size; i++) {
    int batch_idx = i / (feature_dims.at(0) * vector_size);
    int out_i = i % (feature_dims.at(0) * vector_size);
    float diff = std::abs(static_cast<float>(host_in_0_grad_ptr[i]) -
                          static_cast<float>(in_0_grad_data.at(i)));
    if (diff > 0.001) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff
                << " out0: " << static_cast<float>(host_in_0_grad_ptr[i])
                << " out1 " << static_cast<float>(in_0_grad_data.at(i))
                << std::endl;
    }
  }

  std::ifstream in_1_grad_is;
  in_1_grad_is.open("in_1_grad.bin");
  std::vector<T> in_1_grad_data(batch_size * feature_dims.at(1) * vector_size);
  in_1_grad_is.read(reinterpret_cast<char *>(in_1_grad_data.data()), in_1_size);

  for (int i = 0; i < batch_size * feature_dims.at(1) * vector_size; i++) {
    int batch_idx = i / (feature_dims.at(1) * vector_size);
    int out_i = i % (feature_dims.at(1) * vector_size);
    float diff = std::abs(static_cast<float>(host_in_1_grad_ptr[i]) -
                          static_cast<float>(in_1_grad_data.at(i)));
    if (diff > 0.001) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff
                << " out0: " << static_cast<float>(host_in_1_grad_ptr[i])
                << " out1 " << static_cast<float>(in_1_grad_data.at(i))
                << std::endl;
    }
  }

  std::ifstream output_concat_grad_is;
  output_concat_grad_is.open("output_concat_grad.bin");
  std::vector<T> output_concat_grad_data(batch_size * vector_size);
  output_concat_grad_is.read(
      reinterpret_cast<char *>(output_concat_grad_data.data()), in_0_size);

  for (int i = 0; i < batch_size * vector_size; i++) {
    int batch_idx = i / (vector_size);
    int out_i = i % (vector_size);
    float diff = std::abs(static_cast<float>(host_output_concat_grad_ptr[i]) -
                          static_cast<float>(output_concat_grad_data.at(i)));
    if (diff > 0.001) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff << " out0: "
                << static_cast<float>(host_output_concat_grad_ptr[i])
                << " out1 " << static_cast<float>(output_concat_grad_data.at(i))
                << std::endl;
    }
  }
  CudaCheck(cudaFree(in_0_ptr));
  CudaCheck(cudaFreeHost(host_in_0_ptr));
  CudaCheck(cudaFreeHost(host_in_1_ptr));
  CudaCheck(cudaFree(in_1_ptr));
  CudaCheck(cudaFree(dy_ptr));
  CudaCheck(cudaFreeHost(host_dy_ptr));
  return 0;
}
