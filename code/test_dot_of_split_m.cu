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

template <typename T, int32_t N> struct DotFwdParam {
  const T *in[N];
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  const T *output_concat;
  int32_t features_dim;
  T *out;
  int32_t num_in;
};

template <typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

constexpr int unroll_dim = 2;

template <int32_t N, int32_t pack_size, int TILE_DIM, int M_BLOCKS,
          int K_BLOCKS>
__global__ void DotFeatureInteractionHalf(
    int64_t batch_size, int padded_num_rows, int vector_num_pack,
    int out_num_cols, int out_num_cols_num_pack, int in_shared_mem_cols,
    int in_shared_mem_cols_num_pack, int acc_shared_mem_cols,
    int acc_shared_mem_cols_num_pack, int warp_shared_mem_bytes, int offset,
    int output_padding, DotFwdParam<half, N> param) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  half *buf = reinterpret_cast<half *>(shared_buf);
  Pack<half, pack_size> *buf_pack =
      reinterpret_cast<Pack<half, pack_size> *>(shared_buf);
  float *acc_buf = reinterpret_cast<float *>(
      shared_buf + padded_num_rows * in_shared_mem_cols * sizeof(half));
  int batch_idx = blockIdx.x;
  half *batch_out = param.out + batch_idx * out_num_cols;
  Pack<half, pack_size> *batch_out_pack =
      reinterpret_cast<Pack<half, pack_size> *>(param.out) +
      batch_idx * out_num_cols_num_pack;
  const int output_concat_size = 128;
  const half *batch_output_concat =
      param.output_concat + batch_idx * output_concat_size;
  for (int col = threadIdx.x; col < vector_num_pack; col += blockDim.x) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (i >= param.num_in) {
        break;
      }
      const Pack<half, pack_size> *batch_in =
          reinterpret_cast<const Pack<half, pack_size> *>(param.in[i]) +
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
          buf_pack[buf_row * in_shared_mem_cols_num_pack + col] =
              batch_in[in_row * vector_num_pack + col];
        }
      }
    }
  }
  __syncthreads(); 
  for (int blocks_id = warp_id; blocks_id < M_BLOCKS * M_BLOCKS;
       blocks_id += blockDim.y) {
    int blocks_row_id = blocks_id / M_BLOCKS;
    int blocks_col_id = blocks_id % M_BLOCKS;
    if (blocks_row_id >= blocks_col_id) {
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM,
                             TILE_DIM, float>
          acc;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM,
                             TILE_DIM, half, nvcuda::wmma::row_major>
          a;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM,
                             TILE_DIM, half, nvcuda::wmma::col_major>
          b;
      nvcuda::wmma::fill_fragment(acc, 0.0f);
      for (int step = 0; step < K_BLOCKS; ++step) {
        half *tile_a_ptr = buf + blocks_row_id * TILE_DIM * in_shared_mem_cols +
                           step * TILE_DIM;
        half *tile_b_ptr = buf + blocks_col_id * TILE_DIM * in_shared_mem_cols +
                           step * TILE_DIM;
        nvcuda::wmma::load_matrix_sync(a, tile_a_ptr, in_shared_mem_cols);
        nvcuda::wmma::load_matrix_sync(b, tile_b_ptr, in_shared_mem_cols);
        nvcuda::wmma::mma_sync(acc, a, b, acc);
      }
      float *tile_ptr = acc_buf +
                        blocks_row_id * TILE_DIM * acc_shared_mem_cols +
                        blocks_col_id * TILE_DIM;
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc, acc_shared_mem_cols,
                                      nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  half *emb_out = reinterpret_cast<half *>(batch_out_pack + vector_num_pack);
  for (int base_row = threadIdx.y * unroll_dim; base_row < param.features_dim;
       base_row += unroll_dim * blockDim.y) {
#pragma unroll
    for (int k = 0; k < unroll_dim; ++k) {
      int row = base_row + k;
      if (row >= param.features_dim) {
        break;
      }
      for (int col = threadIdx.x; col < param.features_dim; col += blockDim.x) {
        if (col < row + offset) {
          int64_t idx = row * (offset + row - 1 + offset) / 2 + col;
          emb_out[idx] = __float2half(acc_buf[row * acc_shared_mem_cols + col]);
        }
      }
    }
  }
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < output_concat_size;
       i += blockDim.x * blockDim.y) {
    batch_out[i] = batch_output_concat[i];
  }
  for (int i = thread_id; i < output_padding; i += blockDim.x * blockDim.y) {
    batch_out[out_num_cols - 1 - i] = 0;
  }
}

int main() {
  using T = half; // int
  int64_t batch_size = 55296 / 8;
  int64_t vector_size = 128;
  int64_t out_num_cols = 480;
  std::vector<int64_t> feature_dims = {1, 26};
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
  T *host_out_ptr;
  T *out_ptr;
  int64_t out_dim = 480;
  size_t out_size = batch_size * out_dim * sizeof(T);
  CudaCheck(cudaMalloc(&out_ptr, out_size));
  CudaCheck(cudaMallocHost(&host_out_ptr, out_size));

  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  std::ifstream in_0_is;
  in_0_is.open("/data/guoran/data/in_0.bin");
  in_0_is.read(reinterpret_cast<char *>(host_in_0_ptr), in_0_size);
  CudaCheck(cudaMemcpy(in_0_ptr, host_in_0_ptr, in_0_size, cudaMemcpyDefault));

  std::ifstream in_1_is;
  in_1_is.open("/data/guoran/data/in_1.bin");
  in_1_is.read(reinterpret_cast<char *>(host_in_1_ptr), in_1_size);
  CudaCheck(cudaMemcpy(in_1_ptr, host_in_1_ptr, in_1_size, cudaMemcpyDefault));

  const int pack_size = 4;
  const int TILE_DIM = 16;
  const int M_BLOCKS = 2;
  const int K_BLOCKS = 8;
  const int skew_half = 8;
  const int skew_acc = 8;
  const int concated_padded_dim = 32;
  const int in_shared_mem_num_cols = vector_size + skew_half;
  const int acc_shared_mem_num_cols = concated_padded_dim + skew_acc;
  const size_t in_shared_mem_bytes =
      concated_padded_dim * in_shared_mem_num_cols * sizeof(T);
  const size_t acc_shared_mem_bytes =
      concated_padded_dim * acc_shared_mem_num_cols * sizeof(float);
  const size_t warp_shared_mem_bytes =
      in_shared_mem_bytes + acc_shared_mem_bytes;
  const int block_size = 128;
  const int block_dim_x = 32;
  const int block_dim_y = block_size / block_dim_x;
  const int num_blocks = batch_size;

  const int out_num_cols_num_pack = out_num_cols / pack_size;
  const int vector_num_pack = vector_size / pack_size;
  const int in_shared_mem_cols_num_pack = in_shared_mem_num_cols / pack_size;
  const int acc_shared_mem_cols_num_pack = acc_shared_mem_num_cols / pack_size;

  DotFwdParam<T, 2> param;
  param.in[0] = in_0_ptr;
  param.in[1] = in_1_ptr;
  param.in_feature_dim[0] = feature_dims.at(0);
  param.in_feature_dim[1] = feature_dims.at(1);
  param.dim_start_offset[0] = 0;
  param.dim_start_offset[1] = feature_dims.at(0);
  param.num_in = 2;
  param.out = out_ptr;
  param.features_dim = 27;
  param.output_concat = in_0_ptr;

  DotFeatureInteractionHalf<2, 4, TILE_DIM, M_BLOCKS, K_BLOCKS>
      <<<num_blocks, dim3(block_dim_x, block_dim_y), warp_shared_mem_bytes,
         stream>>>(batch_size, concated_padded_dim, vector_num_pack,
                   out_num_cols, out_num_cols_num_pack, in_shared_mem_num_cols,
                   in_shared_mem_cols_num_pack, acc_shared_mem_num_cols,
                   acc_shared_mem_cols_num_pack, warp_shared_mem_bytes, 0, 1,
                   param);
  CudaCheck(cudaMemcpy(host_out_ptr, out_ptr, out_size, cudaMemcpyDefault));

  std::ifstream out_is;
  out_is.open("/data/guoran/data/out.bin");
  std::vector<half> out_data(batch_size * out_dim);
  out_is.read(reinterpret_cast<char *>(out_data.data()), out_size);

  for (int i = 0; i < batch_size * out_dim; i++) {
    int batch_idx = i / out_dim;
    int out_i = i % out_dim;
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
