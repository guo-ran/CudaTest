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
  T *out;
  int32_t num_in;
};

template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

constexpr int padded_num_rows = 32;
constexpr int skew_half = 8; // for align and bank conflict
constexpr int shared_mem_num_cols = 128 + skew_half;
constexpr int shared_mem_num_cols_pack4 = shared_mem_num_cols / 4;
constexpr int shared_mem_num_cols_acc = 32 + skew_half;
constexpr int TILE_DIM = 16;
constexpr int M_BLOCKS = 2;
constexpr int K_BLOCKS = 8;
constexpr int shared_mem_stride_bytes =
    padded_num_rows * shared_mem_num_cols * sizeof(half);
constexpr int out_num_cols = 480;
constexpr int out_num_cols_pack4 = out_num_cols / 4;

template <int32_t N>
__global__ void DotFeatureInteraction(int64_t batch_size,
                                      int64_t embedding_size,
                                      int64_t embedding_num_pack,
                                      Param<half, N> param) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  half *buf =
      reinterpret_cast<half *>(shared_buf + warp_id * shared_mem_stride_bytes);
  Pack<half, 4> *buf_pack4 =
      reinterpret_cast<Pack<half, 4> *>(shared_buf + warp_id * shared_mem_stride_bytes);
  float *acc_buf =
      reinterpret_cast<float *>(shared_buf + warp_id * shared_mem_stride_bytes);
  int global_warp_id = warp_id + blockDim.y * blockIdx.x;
  for (int batch_idx = global_warp_id; batch_idx < batch_size;
       batch_idx += blockDim.y * gridDim.x) {
    half *batch_out = param.out + batch_idx * out_num_cols;
    Pack<half, 4>  *batch_out_pack4 = reinterpret_cast<Pack<half, 4> *>(param.out) + batch_idx * out_num_cols_pack4;
    // 1. load in to shared
    int row = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      //if (i >= param.num_in) {
      //  break;
      //}
      //const half *batch_in =
      //    param.in[i] + batch_idx * param.in_feature_dim[i] * embedding_size;
      const Pack<half, 4>* batch_in = reinterpret_cast<const Pack<half, 4>*>(param.in[i]) + batch_idx * param.in_feature_dim[i] * embedding_num_pack;
      for (int j = 0; j < param.in_feature_dim[i]; ++j) {
        for (int col = threadIdx.x; col < embedding_num_pack; col += blockDim.x) {
          buf_pack4[row * shared_mem_num_cols_pack4 + col] =
              batch_in[j * embedding_num_pack + col];
        }
        row++;
      }
    }
    Pack<half, 4> zero;
    for(int k=0;k<4;++k) {
      zero.elem[k] = 0;
    }
    for (int i = row; i < padded_num_rows; ++i) {
      for (int col = threadIdx.x; col < embedding_num_pack; col += blockDim.x) {
        buf_pack4[i * shared_mem_num_cols_pack4 + col] = zero;
      }
    }
    for (int col = threadIdx.x; col < embedding_num_pack; col += blockDim.x) {
      batch_out_pack4[col] = buf_pack4[col];
    }
    // 2. load to tensor core
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM,
                           TILE_DIM, float>
        acc[M_BLOCKS][M_BLOCKS];
    for (int i = 0; i < M_BLOCKS; ++i) {
      for (int j = 0; j < M_BLOCKS; ++j) {
        nvcuda::wmma::fill_fragment(acc[i][j], 0.0f);
      }
    }
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::row_major>
        a[M_BLOCKS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::col_major>
        b[M_BLOCKS];
    for (int step = 0; step < K_BLOCKS; ++step) {
      for (int j = 0; j < M_BLOCKS; ++j) {
        half *tile_ptr =
            buf + j * TILE_DIM * shared_mem_num_cols + step * TILE_DIM;
        nvcuda::wmma::load_matrix_sync(a[j], tile_ptr, shared_mem_num_cols);
        nvcuda::wmma::load_matrix_sync(b[j], tile_ptr, shared_mem_num_cols);
      }
      for (int i = 0; i < M_BLOCKS; ++i) {
        for (int j = 0; j < M_BLOCKS; ++j) {
          if (i < j) {
            continue;
          }
          nvcuda::wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
        if (i < j) {
          continue;
        }
        float *tile_ptr =
            acc_buf + i * TILE_DIM * shared_mem_num_cols_acc + j * TILE_DIM;
        nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i][j],
                                        shared_mem_num_cols_acc,
                                        nvcuda::wmma::mem_row_major);
      }
    }

    half *emb_out = batch_out + embedding_size;
    for (int row = 0; row < M_BLOCKS * TILE_DIM; ++row) {
      for (int col = threadIdx.x; col < M_BLOCKS * TILE_DIM;
           col += blockDim.x) {
        if (col < row) {
          uint offset = (row * (row - 1)) / 2 + col;
          emb_out[offset] =
              __float2half(acc_buf[row * shared_mem_num_cols_acc + col]);
        }
      }
    }
    if (threadIdx.x == 0) {
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
  int64_t vector_num_pack = vector_size / 4;
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
  in_0_is.open("in_0.bin");
  in_0_is.read(reinterpret_cast<char *>(host_in_0_ptr), in_0_size);
  CudaCheck(cudaMemcpy(in_0_ptr, host_in_0_ptr, in_0_size, cudaMemcpyDefault));

  std::ifstream in_1_is;
  in_1_is.open("in_1.bin");
  in_1_is.read(reinterpret_cast<char *>(host_in_1_ptr), in_1_size);
  CudaCheck(cudaMemcpy(in_1_ptr, host_in_1_ptr, in_1_size, cudaMemcpyDefault));

  int block_size = 128;
  int block_dim_x = 32;
  int block_dim_y = block_size / block_dim_x;
  int num_blocks = batch_size / block_dim_y;
  Param<T, 2> param;
  param.in[0] = in_0_ptr;
  param.in[1] = in_1_ptr;
  param.in_feature_dim[0] = feature_dims.at(0);
  param.in_feature_dim[1] = feature_dims.at(1);
  param.num_in = 2;
  param.out = out_ptr;
  size_t shared_mem_bytes = block_dim_y * shared_mem_stride_bytes;
  DotFeatureInteraction<2>
      <<<num_blocks, dim3(block_dim_x, block_dim_y), shared_mem_bytes,
         stream>>>(batch_size, vector_size, vector_num_pack, param);

  CudaCheck(cudaMemcpy(host_out_ptr, out_ptr, out_size, cudaMemcpyDefault));

  std::ifstream out_is;
  out_is.open("out.bin");
  std::vector<half> out_data(batch_size * out_dim);
  out_is.read(reinterpret_cast<char *>(out_data.data()), out_size);

  for (int i = 0; i < batch_size * out_dim; i++) {
    float diff = static_cast<float>(host_out_ptr[i]) -
                 static_cast<float>(out_data.at(i));
    if (diff > 0.00001) {
      std::cout << "i " << i << " diff " << diff << std::endl;
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
