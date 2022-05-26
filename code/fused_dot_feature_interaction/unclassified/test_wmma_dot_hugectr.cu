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

template <uint x> struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <> struct Log2<1> { static constexpr uint value = 0; };
struct __align__(8) half4 {
  half2 vals[2];
};

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint M_BLOCKS,
          uint K_BLOCKS, uint SMEM_STRIDE, uint SMEM_STRIDE_ACC,
          uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2, uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractFwdKernelNonAligned(
        const __half *__restrict bottom_mlp_input,
        const __half *__restrict emb_input, __half *__restrict output,
        uint batch_size, uint num_rows, uint num_cols,
        uint num_rows_after_padding, uint num_cols_after_padding,
        uint smem_elems_per_warp, uint smem_rows_per_warp, uint output_size,
        uint num_row_steps, uint num_col_steps) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (THREADS_IN_WARP - 1);

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

  // const half *sample_input = input + num_rows * num_cols * sample_id;
  const half *sample_bottom_mlp_input = bottom_mlp_input + num_cols * sample_id;
  const half *sample_emp_input =
      emb_input + (num_rows - 1) * num_cols * sample_id;
  const half *sample_input = sample_bottom_mlp_input;
  // for (uint i = 0; i < num_rows; ++i, sample_input += num_cols) {
  for (uint i = 0; i < num_rows; ++i) {
    for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
      (shmem + i * SMEM_STRIDE)[idx] = sample_input[idx];
    }
    sample_input = (i == 0) ? sample_emp_input : (sample_input + num_cols);
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
    }
  }
  __syncwarp();
  half *gmem_output = output + output_size * sample_id;

  for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
    gmem_output[idx] = shmem[idx];
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM,
                         TILE_DIM, float>
      acc[M_BLOCKS][M_BLOCKS];

  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      nvcuda::wmma::fill_fragment(acc[i][j], 0);
    }
  }

  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::row_major>
        a[M_BLOCKS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::col_major>
        b[M_BLOCKS];
    for (int j = 0; j < M_BLOCKS; j++) {
      int base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
      const half *tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
      nvcuda::wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
      nvcuda::wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
        nvcuda::wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC,
                                      nvcuda::wmma::mem_row_major);
    }
  }

  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] =
          __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
    }
  }
  // Padding
  if (lane_id == 0) {
    gmem_output[output_size - 1] = __float2half(0);
  }
#else
#warning                                                                       \
    "dotBasedInteractFwdKernelNonAligned is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint M_BLOCKS,
          uint K_BLOCKS, uint SMEM_STRIDE, uint SMEM_STRIDE_ACC,
          uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2, uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractFwdKernel(const __half *__restrict bottom_mlp_input,
                                   const __half *__restrict emb_input,
                                   __half *__restrict output, uint batch_size,
                                   uint num_rows, uint num_cols,
                                   uint num_rows_after_padding,
                                   uint num_cols_after_padding,
                                   uint smem_elems_per_warp,
                                   uint smem_rows_per_warp, uint output_size,
                                   uint num_row_steps, uint num_col_steps) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  int sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  int lane_id = threadIdx.x & (THREADS_IN_WARP - 1);

  extern __shared__ half shmem_dynamic[];
  half *shmem = shmem_dynamic + (warp_id * smem_elems_per_warp);

  // const half *sample_input = input + num_rows * num_cols * sample_id;
  const half *sample_bottom_mlp_input = bottom_mlp_input + num_cols * sample_id;
  const half *sample_emp_input =
      emb_input + (num_rows - 1) * num_cols * sample_id;
  const half *sample_input = sample_bottom_mlp_input;
  if (lane_id < (num_cols >> 2)) {
    // for (int i = 0; i < num_rows; ++i, sample_input += num_cols) {
    for (int i = 0; i < num_rows; ++i) {
      ((float2 *)(shmem + i * SMEM_STRIDE))[lane_id] =
          ((float2 *)sample_input)[lane_id];
      sample_input = (i == 0) ? sample_emp_input : (sample_input + num_cols);
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (int i = 0; i < num_rows; ++i) {
      (shmem + i * SMEM_STRIDE)[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
    for (int i = num_rows; i < num_rows_after_padding; i++) {
      ((half4 *)(shmem + i * SMEM_STRIDE))[lane_id] = zeros;
    }
  }
  __syncwarp();
  half *gmem_output = output + output_size * sample_id;
  if (lane_id < (num_cols >> 2)) {
    ((float2 *)gmem_output)[lane_id] = ((float2 *)shmem)[lane_id];
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM,
                         TILE_DIM, float>
      acc[M_BLOCKS][M_BLOCKS];

  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
       if (i < j) {
        continue;
      }
      nvcuda::wmma::fill_fragment(acc[i][j], 0);
    }
  }

  for (int k_step = 0; k_step < num_col_steps; k_step++) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::row_major>
        a[M_BLOCKS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM,
                           half, nvcuda::wmma::col_major>
        b[M_BLOCKS];
    for (int j = 0; j < M_BLOCKS; j++) {
      int base_row = (j < M_BLOCKS - 1) ? j * 16 : smem_rows_per_warp - 16;
      const half *tile_ptr = shmem + (base_row * SMEM_STRIDE + k_step * 16);
      nvcuda::wmma::load_matrix_sync(a[j], tile_ptr, SMEM_STRIDE);
      nvcuda::wmma::load_matrix_sync(b[j], tile_ptr, SMEM_STRIDE);
    }
    for (int i = 0; i < M_BLOCKS; i++) {
      for (int j = 0; j < M_BLOCKS; j++) {
         if (i < j) {
          continue;
        }
        nvcuda::wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
  }
  float *shmem_store = reinterpret_cast<float *>(shmem);
  for (int i = 0; i < M_BLOCKS; i++) {
    for (int j = 0; j < M_BLOCKS; j++) {
      if (i < j) {
         continue;
      }
      float *tile_ptr = shmem_store + (i * 16 * SMEM_STRIDE_ACC + j * 16);
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i][j], SMEM_STRIDE_ACC,
                                      nvcuda::wmma::mem_row_major);
    }
  }

  half *gmem_interact_output = gmem_output + num_cols;
  int lastRowBlockOffset = M_BLOCKS * 16 - smem_rows_per_warp;
  int srcLine = 0;
  for (int i = 0; i < num_rows; ++i, ++srcLine) {
    if (i == ((M_BLOCKS - 1) * 16)) {
      srcLine += lastRowBlockOffset;
    }
    if (lane_id < i) {
      uint offset = (i * (i - 1)) >> 1;
      gmem_interact_output[offset + lane_id] =
          __float2half(shmem_store[srcLine * SMEM_STRIDE_ACC + lane_id]);
    }
  }
  // Padding
  if (lane_id == 0) {
    gmem_output[output_size - 1] = __float2half(0);
  }
#else
#warning                                                                       \
    "dotBasedInteractFwdKernel is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

inline void dotBasedInteractFwd(const void *bottom_mlp_input,
                                const void *emb_input, void *output,
                                uint batch_size, uint num_rows, uint num_cols,
                                cudaStream_t stream) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint warps_per_threadblock = 4;
  const uint threadblock_size = warps_per_threadblock * 32;
  const uint kPaddingSize = 1;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  uint num_row_steps = num_row_tiles / kRowTilesPerStep;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  const uint K_BLOCKS = 8;
  const uint M_BLOCKS = 2;
  const uint SKEW_HALF = ((K_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE = (K_BLOCKS * 16 + SKEW_HALF);
  // multiple of 2 to guarantee 256-bit alignment for start of the row, at least
  // 16 to safeload a tile
  const uint smem_rows_per_warp = M_BLOCKS << 4;
  const uint smem_elems_per_warp_mat = smem_rows_per_warp * SMEM_STRIDE;
  const uint SKEW_HALF_ACC = ((M_BLOCKS % 2) == 0) ? 8 : 0;
  const uint SMEM_STRIDE_ACC = (M_BLOCKS * 16 + SKEW_HALF_ACC);
  const uint smem_elems_per_warp_acc =
      M_BLOCKS * 16 * SMEM_STRIDE_ACC * 2; // output in FP32
  const uint smem_elems_per_warp =
      (smem_elems_per_warp_mat > smem_elems_per_warp_acc)
          ? smem_elems_per_warp_mat
          : smem_elems_per_warp_acc;
  uint output_size = num_cols + (num_rows * (num_rows - 1) >> 1) + kPaddingSize;
  std::cout<<"smem_rows_per_warp "<<smem_rows_per_warp<<std::endl;
  std::cout<<"smem_elems_per_warp_mat "<<smem_elems_per_warp_mat<<std::endl;
  std::cout<<"SKEW_HALF "<<SKEW_HALF<<std::endl;
  std::cout<<"SKEW_HALF_ACC "<<SKEW_HALF_ACC<<std::endl;
  std::cout<<"SMEM_STRIDE "<<SMEM_STRIDE<<std::endl;
  std::cout<<"SMEM_STRIDE_ACC "<<SMEM_STRIDE_ACC<<std::endl;
  std::cout<<"smem_elems_per_warp "<<smem_elems_per_warp<<std::endl;
  bool float4_predicate = !((num_cols & 7) || (output_size & 7));

  if (float4_predicate) {
    dotBasedInteractFwdKernel<warps_per_threadblock, threadblock_size, M_BLOCKS,
                              K_BLOCKS, SMEM_STRIDE, SMEM_STRIDE_ACC, kWarpSize,
                              kWarpSizeLog2, kTileDim, kTileDimLog2>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock,
           threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half),
           stream>>>(
            (const __half *)bottom_mlp_input, (const __half *)emb_input,
            (half *)output, batch_size, num_rows, num_cols,
            num_rows_after_padding, num_cols_after_padding, smem_elems_per_warp,
            smem_rows_per_warp, output_size, num_row_steps, num_col_steps);
  } else {
    dotBasedInteractFwdKernelNonAligned<warps_per_threadblock, threadblock_size,
                                        M_BLOCKS, K_BLOCKS, SMEM_STRIDE,
                                        SMEM_STRIDE_ACC, kWarpSize,
                                        kWarpSizeLog2, kTileDim, kTileDimLog2>
        <<<(batch_size + warps_per_threadblock - 1) / warps_per_threadblock,
           threadblock_size,
           warps_per_threadblock * smem_elems_per_warp * sizeof(__half),
           stream>>>(
            (const __half *)bottom_mlp_input, (const __half *)emb_input,
            (half *)output, batch_size, num_rows, num_cols,
            num_rows_after_padding, num_cols_after_padding, smem_elems_per_warp,
            smem_rows_per_warp, output_size, num_row_steps, num_col_steps);
  }
}

int main() {
  using T = half; // int
  int64_t batch_size = 55296;
  int64_t vector_size = 128;
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

  dotBasedInteractFwd(in_0_ptr, in_1_ptr, out_ptr, batch_size, 27, vector_size,
                      stream);

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
