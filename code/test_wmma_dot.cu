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

template <typename T, int32_t N> struct Param {
  const T *in[N];
  int32_t in_feature_dim[N];
  int32_t dim_start_offset[N];
  T *out;
  int32_t num_in;
};

template <typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

constexpr int num_warp_per_block = 4;
constexpr int padded_num_rows = 32;
constexpr int skew_half = 8;     // for align and bank conflict
constexpr int skew_half_acc = 8; // for align and bank conflict
constexpr int shared_mem_num_cols = 128 + skew_half;
constexpr int shared_mem_num_cols_pack4 = shared_mem_num_cols / 4;
constexpr int shared_mem_num_cols_acc = 32 + skew_half_acc;
constexpr int in_shared_mem_bytes =
    padded_num_rows * shared_mem_num_cols * sizeof(half);
constexpr int acc_shared_mem_stride_bytes =
    padded_num_rows * shared_mem_num_cols_acc * sizeof(float);
constexpr int acc_shared_mem_bytes = acc_shared_mem_stride_bytes;
constexpr int TILE_DIM = 16;
constexpr int M_BLOCKS = 2;
constexpr int K_BLOCKS = 8;
constexpr int out_num_cols = 480;
constexpr int out_num_cols_pack4 = out_num_cols / 4;
constexpr int num_step = 8;
constexpr int NUM_STEPS_PER_WARP = num_step / num_warp_per_block;
constexpr int unroll_dim = 4;
//每个warp处理2个step. 8个step就是4个warp
//每个warp一个acc buf.

template <int32_t N>
__global__ void DotFeatureInteraction(int batch_size, int embedding_size,
                                      int embedding_num_pack,
                                      Param<half, N> param, half* output_concat) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  half *buf = reinterpret_cast<half *>(shared_buf);
  Pack<half, 4> *buf_pack4 = reinterpret_cast<Pack<half, 4> *>(shared_buf);
  float *acc_buf = reinterpret_cast<float *>(shared_buf);
  for (int batch_idx = blockIdx.x; batch_idx < batch_size;
       batch_idx += gridDim.x) {
    half *batch_out = param.out + batch_idx * out_num_cols;
    Pack<half, 4> *batch_out_pack4 =
        reinterpret_cast<Pack<half, 4> *>(param.out) +
        batch_idx * out_num_cols_pack4;
    const Pack<half, 4>* batch_output_concat = reinterpret_cast<const Pack<half, 4>*>(output_concat) + batch_idx * embedding_num_pack;
    
    const Pack<half, 4> *batch_in_0 =
        reinterpret_cast<const Pack<half, 4> *>(param.in[0]) +
        batch_idx * param.in_feature_dim[0] * embedding_num_pack;
    const Pack<half, 4> *batch_in_1 =
        reinterpret_cast<const Pack<half, 4> *>(param.in[1]) +
        batch_idx * param.in_feature_dim[1] * embedding_num_pack;
    // 1. load in to shared
    Pack<half, 4> zero;
    for (int k = 0; k < 4; ++k) {
      zero.elem[k] = 0;
    }
    for (int row = threadIdx.y * unroll_dim; row < 27;
         row += unroll_dim * blockDim.y) {
      const Pack<half, 4> *batch_in;
#pragma unroll
      for (int k = 0; k < unroll_dim; ++k) {
        int row_id = row + k;
        if (row_id >= 27) {
          break;
        }
        if (row_id == 0) {
          batch_in = batch_in_0;
        } else {
          batch_in = batch_in_1 + (row_id - 1) * embedding_num_pack;
        }
        int col = threadIdx.x;
        buf_pack4[row_id * shared_mem_num_cols_pack4 + col] = batch_in[col];
      }
    }
#pragma unroll
    for (int i = threadIdx.y; i < 5; i += blockDim.y) {
      int row = 27 + i;
      int col = threadIdx.x;
      buf_pack4[row * shared_mem_num_cols_pack4 + col] = zero;
    }
    __syncthreads();// if no this thread sync, error result
    if (warp_id == 1) {
      for (int col = threadIdx.x; col < embedding_num_pack; col += blockDim.x) {
        batch_out_pack4[col] = batch_output_concat[col];//buf_pack4[col];
      }
    }
    // 2. load to tensor core
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM,
                           TILE_DIM, float>
        acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::row_major>
        a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::col_major>
        b;

    for (int step = 0; step < num_step; ++step) {
      int i = warp_id / M_BLOCKS;
      int j = warp_id % M_BLOCKS;
      half *tile_a_ptr =
          buf + i * TILE_DIM * shared_mem_num_cols + step * TILE_DIM;
      half *tile_b_ptr =
          buf + j * TILE_DIM * shared_mem_num_cols + step * TILE_DIM;
      nvcuda::wmma::load_matrix_sync(a, tile_a_ptr, shared_mem_num_cols);
      nvcuda::wmma::load_matrix_sync(b, tile_b_ptr, shared_mem_num_cols);
      nvcuda::wmma::mma_sync(acc, a, b, acc);
    }
    __syncthreads(); // if no this thread sync, error result
    int i = warp_id / M_BLOCKS;
    int j = warp_id % M_BLOCKS;
    float *tile_ptr =
        acc_buf + i * TILE_DIM * shared_mem_num_cols_acc + j * TILE_DIM;
    nvcuda::wmma::store_matrix_sync(tile_ptr, acc, shared_mem_num_cols_acc,
                                    nvcuda::wmma::mem_row_major);
    half *emb_out = batch_out + embedding_size;
    for (int base_row = threadIdx.y * unroll_dim;
         base_row < 27; base_row += unroll_dim * blockDim.y) {
#pragma unroll
      for (int k = 0; k < unroll_dim; ++k) {
        int row = base_row + k;
        if(row>=27) {break;}
        for (int col = threadIdx.x; col < 27;
             col += blockDim.x) {
          if (col < row) {
            uint offset = (row * (row - 1)) / 2 + col;
            emb_out[offset] =
                __float2half(acc_buf[row * shared_mem_num_cols_acc + col]);
          }
        }
      }
    }
    if (warp_id == 0 && threadIdx.x == 0) {
      batch_out[out_num_cols - 1] = 0;
    }
  }
}
// 32 128
/*
16 16 16 16 16 16 16 16
16 16 16 16 16 16 16 16



*/

int main() {
  using T = half; // int
  int64_t batch_size = 55296 / 8;
  int64_t vector_size = 128;
  int64_t embedding_num_pack = vector_size / 4;
  std::vector<int64_t> feature_dims = {1, 26};
  const int features_concated_dim = 27;
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
  int padding=1;
  int64_t out_dim = vector_size + features_concated_dim * (features_concated_dim - 1) / 2 + padding;
  size_t out_size = batch_size * out_dim * sizeof(T);
  CudaCheck(cudaMalloc(&out_ptr, out_size));
  CudaCheck(cudaMallocHost(&host_out_ptr, out_size));

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

  int block_dim_x = 32;
  int block_dim_y = num_warp_per_block;
  int num_blocks = batch_size;
  Param<T, 2> param;
  param.in[0] = in_0_ptr;
  param.in[1] = in_1_ptr;
  param.in_feature_dim[0] = feature_dims.at(0);
  param.in_feature_dim[1] = feature_dims.at(1);
  param.dim_start_offset[0] = 0;
  param.dim_start_offset[1] = feature_dims.at(0);
  param.num_in = 2;
  param.out = out_ptr;
  size_t shared_mem_bytes = std::max(in_shared_mem_bytes, acc_shared_mem_bytes);
  DotFeatureInteraction<2>
      <<<num_blocks, dim3(block_dim_x, block_dim_y), shared_mem_bytes,
         stream>>>(batch_size, vector_size, embedding_num_pack, param, in_0_ptr);

  CudaCheck(cudaMemcpy(host_out_ptr, out_ptr, out_size, cudaMemcpyDefault));

  std::ifstream out_is;
  out_is.open("out.bin");
  std::vector<half> out_data(batch_size * out_dim);
  out_is.read(reinterpret_cast<char *>(out_data.data()), out_size);

  for (int i = 0; i < batch_size * out_dim; i++) {
    int batch_idx = i / out_dim;
    int out_i = i % out_dim;
    float diff = static_cast<float>(host_out_ptr[i]) -
                 static_cast<float>(out_data.at(i));
    if (diff > 0.01) {
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
