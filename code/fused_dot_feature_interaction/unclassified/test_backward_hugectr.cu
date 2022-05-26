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

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <>
struct Log2<1> {
  static constexpr uint value = 0;
};

struct __align__(8) half4 {
  half2 vals[2];
};

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP, uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2, uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void dotBasedInteractBwdKernelNonAligned(
    const __half *__restrict upstream_grad, half __restrict *bottom_mlp_grad,
    half __restrict *emb_grad, uint batch_size, uint num_rows, uint num_cols,
    uint num_rows_after_padding, uint num_cols_after_padding, uint sample_size,
    uint interaction_ugrad_size, uint interaction_ugrad_size_with_padding,
    uint interaction_ugrad_2D_size_elems, uint interaction_ugrad_2D_stride, uint input_size_elems,
    uint input_stride, uint num_row_steps, uint num_col_steps, uint row_tiles_per_step,
    uint shared_mem_per_warp_size_byte) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  extern __shared__ half shared_mem[];
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  uint lane_id = threadIdx.x & (THREADS_IN_WARP - 1);
  // ">> 1" to convert to half pointer
  uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);

  half *smem_in = &shared_mem[smem_warp_offset];
  half *smem_temp = &shared_mem[smem_warp_offset + input_size_elems];
  float *smem_out = reinterpret_cast<float *>(smem_temp);

  // Global memory pointers for the current sample
  // Input
  // uint gmem_input_sample_offset = sample_id * sample_size;
  // const half *gmem_input = &input[gmem_input_sample_offset];
  uint gmem_bottom_mlp_input_sample_offset = sample_id * num_cols;
  uint gmem_emb_input_sample_offset = sample_id * (num_rows - 1) * num_cols;
  const half *gmem_bottom_mlp_input = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  const half *gmem_emb_input = &emb_grad[gmem_emb_input_sample_offset];

  // Interaction Gradient
  // const uint &gmem_grad_sample_offset = gmem_input_sample_offset;
  // half *gmem_grad = &grad[gmem_grad_sample_offset];
  half *gmem_bottom_mlp_grad = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  half *gmem_emb_grad = &emb_grad[gmem_emb_input_sample_offset];

  // Bottom MLP gradient
  // half *gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

  // Upstream gradient vector
  uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
  const half *gmem_ugrad = &upstream_grad[gmem_ugrad_sample_offset];

  // Upstream gradient vector for interactions
  const half *gmem_ugrad_interactions = &gmem_ugrad[num_cols];

// upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
  for (uint idx = lane_id; idx < interaction_ugrad_size; idx += THREADS_IN_WARP) {
    smem_in[idx] = gmem_ugrad_interactions[idx];
  }
  __syncwarp();
  // Form the 2D ugrad matrix.
  if (lane_id < num_rows_after_padding) {
    uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
    uint ugrad_offset_1 = lane_id * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      half ugrad_val = __float2half(0.0f);
      if (row < lane_id && lane_id < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_temp[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= lane_id && lane_id < num_rows_after_padding) {
        smem_temp[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_temp[row * interaction_ugrad_2D_stride + lane_id] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Input -> Shared Memory

  for (uint row = 0; row < num_rows; row++) {
    half *smem_row_ptr = &smem_in[row * input_stride];
    // const half *gmem_row_ptr = &gmem_input[row * num_cols];
    const half *gmem_row_ptr =
        (row == 0) ? gmem_bottom_mlp_input : &gmem_emb_input[(row - 1) * num_cols];
    for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
      smem_row_ptr[idx] = gmem_row_ptr[idx];
    }
    uint idx = lane_id + num_cols;
    if (idx < num_cols_after_padding) {
      smem_row_ptr[idx] = __float2half(0);
    }
  }

#pragma unroll 2
  for (uint row = num_rows; row < num_rows_after_padding; row++) {
    half *smem_row_ptr = &smem_in[row * input_stride];
    for (uint idx = lane_id; idx < num_cols_after_padding; idx += THREADS_IN_WARP) {
      smem_row_ptr[idx] = __float2half(0);
    }
  }
  __syncwarp();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      a[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];
  for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
      const half *tile_ptr = smem_temp + ((i * interaction_ugrad_2D_stride + j) << TILE_DIM_LOG_2);
      nvcuda::wmma::load_matrix_sync(a[i][j], tile_ptr, interaction_ugrad_2D_stride);
    }
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>
      acc[ROW_TILES_PER_STEP];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      b[ROW_TILES_PER_STEP];
  for (int col_step = 0; col_step < num_col_steps; col_step++) {
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      const half *tile_ptr = smem_in + ((i * input_stride + col_step) << TILE_DIM_LOG_2);
      nvcuda::wmma::fill_fragment(acc[i], 0);
      nvcuda::wmma::load_matrix_sync(b[i], tile_ptr, input_stride);
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
        nvcuda::wmma::mma_sync(acc[i], a[i][j], b[j], acc[i]);
      }
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      float *tile_ptr = smem_out + i * TILE_DIM * TILE_DIM;
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i], TILE_DIM, nvcuda::wmma::mem_row_major);
    }
    __syncwarp();
    uint gmem_grad_col = (col_step << TILE_DIM_LOG_2) + lane_id;
    if (gmem_grad_col < num_cols) {
      for (uint i = 0; i < num_rows; i++) {
        // gmem_grad[i * num_cols + gmem_grad_col] = __float2half(smem_out[(i << TILE_DIM_LOG_2) +
        // lane_id]);
        half *gmem_grad = (i == 0) ? gmem_bottom_mlp_grad : gmem_emb_grad;
        uint idx = (i == 0) ? gmem_grad_col : ((i - 1) * num_cols + gmem_grad_col);
        half val = __float2half(smem_out[(i << TILE_DIM_LOG_2) + lane_id]);
        gmem_grad[idx] = (i == 0) ? (val + gmem_ugrad[idx]) : val;
      }
    }
  }

// for (uint idx = lane_id; idx < num_cols; idx += THREADS_IN_WARP) {
//   gmem_mlp_grad[idx] = gmem_ugrad[idx];
// }
#else
#warning "dotBasedInteractBwdKernelNonAligned is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

template <uint WARPS_PER_BLOCK, uint THREADBLOCK_SIZE, uint ROW_TILES_PER_STEP,
          uint COL_TILES_PER_STEP, uint THREADS_IN_WARP, uint THREADS_IN_WARP_LOG_2, uint TILE_DIM,
          uint TILE_DIM_LOG_2>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void dotBasedInteractBwdKernel(const __half *__restrict upstream_grad,
                                   half __restrict *bottom_mlp_grad, half __restrict *emb_grad,
                                   uint batch_size, uint num_rows, uint num_cols,
                                   uint num_rows_after_padding, uint num_cols_after_padding,
                                   uint sample_size, uint interaction_ugrad_size,
                                   uint interaction_ugrad_size_with_padding,
                                   uint interaction_ugrad_2D_size_elems,
                                   uint interaction_ugrad_2D_stride, uint input_size_elems,
                                   uint input_stride, uint num_row_steps, uint num_col_steps,
                                   uint row_tiles_per_step, uint shared_mem_per_warp_size_byte) {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
  extern __shared__ half shared_mem[];
  uint warp_id = (threadIdx.x >> THREADS_IN_WARP_LOG_2);
  uint sample_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (sample_id >= batch_size) {
    return;
  }
  uint lane_id = threadIdx.x & (THREADS_IN_WARP - 1);
  // ">> 1" to convert to half pointer
  uint smem_warp_offset = warp_id * (shared_mem_per_warp_size_byte >> 1);

  half *smem_in = &shared_mem[smem_warp_offset];
  half *smem_temp = &shared_mem[smem_warp_offset + input_size_elems];
  float *smem_out = reinterpret_cast<float *>(smem_temp);

  // Global memory pointers for the current sample
  // Input
  // uint gmem_input_sample_offset = sample_id * sample_size;
  // const half *gmem_input = &input[gmem_input_sample_offset];
  uint gmem_bottom_mlp_input_sample_offset = sample_id * num_cols;
  uint gmem_emb_input_sample_offset = sample_id * (num_rows - 1) * num_cols;
  const half *gmem_bottom_mlp_input = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  const half *gmem_emb_input = &emb_grad[gmem_emb_input_sample_offset];

  // Interaction Gradient
  // const uint &gmem_grad_sample_offset = gmem_input_sample_offset;
  // half *gmem_grad = &grad[gmem_grad_sample_offset];
  half *gmem_bottom_mlp_grad = &bottom_mlp_grad[gmem_bottom_mlp_input_sample_offset];
  half *gmem_emb_grad = &emb_grad[gmem_emb_input_sample_offset];

  // Bottom MLP gradient
  // half *gmem_mlp_grad = &bottom_mlp_grad[sample_id * num_cols];

  // Upstream gradient vector
  uint gmem_ugrad_sample_offset = sample_id * (num_cols + interaction_ugrad_size_with_padding);
  const half *gmem_ugrad = &upstream_grad[gmem_ugrad_sample_offset];

  // Upstream gradient vector for interactions
  const half *gmem_ugrad_interactions = &gmem_ugrad[num_cols];

// upstream grad -> shared memory (place in input section temporarily)
#pragma unroll
  for (uint idx = lane_id; idx < (interaction_ugrad_size >> 3); idx += THREADS_IN_WARP) {
    ((float4 *)smem_in)[idx] = ((float4 *)gmem_ugrad_interactions)[idx];
  }
  uint offset = (interaction_ugrad_size >> 3) << 3;
  for (uint idx = lane_id + offset; idx < interaction_ugrad_size; idx += THREADS_IN_WARP) {
    smem_in[idx] = gmem_ugrad_interactions[idx];
  }
  __syncwarp();
  // Form the 2D ugrad matrix.
  if (lane_id < num_rows_after_padding) {
    uint ugrad_flat_index = ((lane_id * (lane_id - 1)) >> 1);
    uint ugrad_offset_1 = lane_id * interaction_ugrad_2D_stride;
    for (uint row = 0; row < num_rows; row++) {
      half ugrad_val = __float2half(0.0f);
      if (row < lane_id && lane_id < num_rows) {
        ugrad_val = smem_in[ugrad_flat_index + row];
        smem_temp[ugrad_offset_1 + row] = ugrad_val;
      }
      if (row <= lane_id && lane_id < num_rows_after_padding) {
        smem_temp[row * interaction_ugrad_2D_stride + lane_id] = ugrad_val;
      }
    }
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      smem_temp[row * interaction_ugrad_2D_stride + lane_id] = __float2half(0.0f);
    }
  }
  __syncwarp();

  // Input -> Shared Memory

  if (lane_id < (num_cols >> 2)) {
    for (uint row = 0; row < num_rows; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      // const half *gmem_row_ptr = &gmem_input[row * num_cols];
      const half *gmem_row_ptr =
          (row == 0) ? gmem_bottom_mlp_input : &gmem_emb_input[(row - 1) * num_cols];
      ((float2 *)smem_row_ptr)[lane_id] = ((float2 *)gmem_row_ptr)[lane_id];
    }
  }

  uint idx = lane_id + num_cols;
  if (idx < num_cols_after_padding) {
    for (uint row = 0; row < num_rows; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      smem_row_ptr[idx] = __float2half(0);
    }
  }

  half4 zeros;
  zeros.vals[0].x = __float2half(0);
  zeros.vals[0].y = __float2half(0);
  zeros.vals[1].x = __float2half(0);
  zeros.vals[1].y = __float2half(0);
  if (lane_id < (num_cols_after_padding >> 2)) {
#pragma unroll 2
    for (uint row = num_rows; row < num_rows_after_padding; row++) {
      half *smem_row_ptr = &smem_in[row * input_stride];
      ((half4 *)smem_row_ptr)[lane_id] = zeros;
    }
  }
  __syncwarp();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      a[ROW_TILES_PER_STEP][ROW_TILES_PER_STEP];
  for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
    for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
      const half *tile_ptr = smem_temp + ((i * interaction_ugrad_2D_stride + j) << TILE_DIM_LOG_2);
      nvcuda::wmma::load_matrix_sync(a[i][j], tile_ptr, interaction_ugrad_2D_stride);
    }
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>
      acc[ROW_TILES_PER_STEP];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,
                         nvcuda::wmma::row_major>
      b[ROW_TILES_PER_STEP];
  for (int col_step = 0; col_step < num_col_steps; col_step++) {
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      const half *tile_ptr = smem_in + ((i * input_stride + col_step) << TILE_DIM_LOG_2);
      nvcuda::wmma::fill_fragment(acc[i], 0);
      nvcuda::wmma::load_matrix_sync(b[i], tile_ptr, input_stride);
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      for (uint j = 0; j < ROW_TILES_PER_STEP; j++) {
        nvcuda::wmma::mma_sync(acc[i], a[i][j], b[j], acc[i]);
      }
    }
    for (uint i = 0; i < ROW_TILES_PER_STEP; i++) {
      float *tile_ptr = smem_out + i * TILE_DIM * TILE_DIM;
      nvcuda::wmma::store_matrix_sync(tile_ptr, acc[i], TILE_DIM, nvcuda::wmma::mem_row_major);
    }
    __syncwarp();
    uint gmem_grad_col_base = (col_step << TILE_DIM_LOG_2);
    uint gmem_grad_col = gmem_grad_col_base + lane_id;
    if (gmem_grad_col < num_cols) {
      if (lane_id < 8) {
        ((__half2 *)(gmem_bottom_mlp_grad + gmem_grad_col_base))[lane_id] =
            __hadd2(__float22half2_rn(((float2 *)smem_out)[lane_id]),
                    ((__half2 *)(gmem_ugrad + gmem_grad_col_base))[lane_id]);
      }
      for (uint i = 0; i < num_rows - 1; i++) {
        half val = __float2half(smem_out[((i + 1) << TILE_DIM_LOG_2) + lane_id]);
        gmem_emb_grad[i * num_cols + gmem_grad_col] = val;
      }
    }
  }
#else
#warning "dotBasedInteractBwdKernel is not supported for SM < 70 (or __CUDA_ARCH__ < 700)"
#endif
}

inline void dotBasedInteractBwd(void *upstream_grad, void *bottom_mlp_grad, void *emb_grad,
                                uint batch_size, uint num_rows, uint num_cols,
                                cudaStream_t stream) {
  const uint kWarpSize = 32;
  const uint kWarpSizeLog2 = Log2<kWarpSize>::value;
  const uint kTileDim = 16;
  const uint kTileDimLog2 = Log2<kTileDim>::value;
  const uint mem_skew_size = 8;
  const uint kPaddingSize = 1;
  const uint kWarpsPerBlock = 4;
  const uint kWarpsPerBlockLog2 = Log2<kWarpsPerBlock>::value;
  const uint kNumThreads = kWarpsPerBlock * kWarpSize;
  const uint kRowTilesPerStep = 2;
  const uint kColTilesPerStep = 1;

  uint row_tiles_per_step = num_rows > kTileDim ? kRowTilesPerStep : 1;

  // num tiles
  uint num_row_tiles = (num_rows + kTileDim - 1) >> kTileDimLog2;
  uint num_col_tiles = (num_cols + kTileDim - 1) >> kTileDimLog2;

  // number of rows and columns after padding
  uint num_rows_after_padding = kTileDim << 1;
  uint num_cols_after_padding = num_col_tiles << kTileDimLog2;

  // 2D ugrad size and stride
  uint interaction_ugrad_2D_stride = num_rows_after_padding + mem_skew_size;
  uint interaction_ugrad_2D_size_elems = num_rows_after_padding * interaction_ugrad_2D_stride;
  uint interaction_ugrad_2D_size_bytes = interaction_ugrad_2D_size_elems * sizeof(half);

  // 1D ugrad size
  uint interaction_ugrad_size = num_rows * (num_rows - 1) >> 1;
  uint interaction_ugrad_size_with_padding = interaction_ugrad_size + kPaddingSize;

  // in_out place size and stride
  uint input_stride = num_cols_after_padding + mem_skew_size;
  uint input_size_elems = num_rows_after_padding * input_stride;
  uint input_size_bytes = input_size_elems * sizeof(half);

  // sample size
  uint sample_size = num_rows * num_cols;

  // output size
  uint output_size_elems = kTileDim * kTileDim * kRowTilesPerStep * kColTilesPerStep;
  uint output_size_bytes = output_size_elems * sizeof(float);

  // staging area size
  uint staging_area_size_bytes = output_size_bytes > interaction_ugrad_2D_size_bytes
                                     ? output_size_bytes
                                     : interaction_ugrad_2D_size_bytes;

  // Shared memory size
  uint shared_mem_per_warp_size_byte = input_size_bytes + staging_area_size_bytes;
  uint shared_mem_size_bytes = kWarpsPerBlock * shared_mem_per_warp_size_byte;

  uint num_blocks = (batch_size + kWarpsPerBlock - 1) >> kWarpsPerBlockLog2;
  uint num_row_steps = num_row_tiles / row_tiles_per_step;
  uint num_col_steps = num_col_tiles / kColTilesPerStep;

  bool float4_predicate = !((interaction_ugrad_size_with_padding & 7) || (num_cols & 7));
  if (float4_predicate) {
    dotBasedInteractBwdKernel<kWarpsPerBlock, kNumThreads, kRowTilesPerStep, kColTilesPerStep,
                              kWarpSize, kWarpSizeLog2, kTileDim, kTileDimLog2>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>(
            (const half *)upstream_grad, (half *)bottom_mlp_grad, (half *)emb_grad, batch_size,
            num_rows, num_cols, num_rows_after_padding, num_cols_after_padding, sample_size,
            interaction_ugrad_size, interaction_ugrad_size_with_padding,
            interaction_ugrad_2D_size_elems, interaction_ugrad_2D_stride, input_size_elems,
            input_stride, num_row_steps, num_col_steps, row_tiles_per_step,
            shared_mem_per_warp_size_byte);
  } else {
    dotBasedInteractBwdKernelNonAligned<kWarpsPerBlock, kNumThreads, kRowTilesPerStep,
                                        kColTilesPerStep, kWarpSize, kWarpSizeLog2, kTileDim,
                                        kTileDimLog2>
        <<<num_blocks, kNumThreads, shared_mem_size_bytes, stream>>>(
            (const half *)upstream_grad, (half *)bottom_mlp_grad, (half *)emb_grad, batch_size,
            num_rows, num_cols, num_rows_after_padding, num_cols_after_padding, sample_size,
            interaction_ugrad_size, interaction_ugrad_size_with_padding,
            interaction_ugrad_2D_size_elems, interaction_ugrad_2D_stride, input_size_elems,
            input_stride, num_row_steps, num_col_steps, row_tiles_per_step,
            shared_mem_per_warp_size_byte);
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

  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  std::ifstream in_0_is;
  in_0_is.open("/data/guoran/data/backward_data/in_0.bin");
  in_0_is.read(reinterpret_cast<char *>(host_in_0_ptr), in_0_size);
  CudaCheck(cudaMemcpy(in_0_grad_ptr, host_in_0_ptr, in_0_size, cudaMemcpyDefault));

  std::ifstream in_1_is;
  in_1_is.open("/data/guoran/data/backward_data/in_1.bin");
  in_1_is.read(reinterpret_cast<char *>(host_in_1_ptr), in_1_size);
  CudaCheck(cudaMemcpy(in_1_grad_ptr, host_in_1_ptr, in_1_size, cudaMemcpyDefault));

  std::ifstream dy_is;
  dy_is.open("/data/guoran/data/backward_data/dy.bin");
  dy_is.read(reinterpret_cast<char *>(host_dy_ptr), out_size);
  CudaCheck(cudaMemcpy(dy_ptr, host_dy_ptr, out_size, cudaMemcpyDefault));

  dotBasedInteractBwd(dy_ptr, in_0_grad_ptr, in_1_grad_ptr, batch_size, 27, vector_size, stream);

  CudaCheck(cudaMemcpy(host_in_0_grad_ptr, in_0_grad_ptr, in_0_size,
                       cudaMemcpyDefault));
  CudaCheck(cudaMemcpy(host_in_1_grad_ptr, in_1_grad_ptr, in_1_size,
                       cudaMemcpyDefault));
  CudaCheck(cudaMemcpy(host_output_concat_grad_ptr, output_concat_grad_ptr,
                       in_0_size, cudaMemcpyDefault));

  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaDeviceSynchronize());
  std::ifstream in_0_grad_is;
  in_0_grad_is.open("/data/guoran/data/backward_data/in_0_grad.bin");
  std::vector<half> in_0_grad_data(batch_size * feature_dims.at(0) *
                                   vector_size);
  in_0_grad_is.read(reinterpret_cast<char *>(in_0_grad_data.data()), in_0_size);

  //hugectr host_in_0_grad_ptr is sum of in_0_grad and concat_output_grad, which is dy, and in this case is 1. 
  for (int i = 0; i < batch_size * feature_dims.at(0) * vector_size; i++) {
    int batch_idx = i / (feature_dims.at(0) * vector_size);
    int out_i = i % (feature_dims.at(0) * vector_size);
    float diff = std::abs(static_cast<float>(host_in_0_grad_ptr[i]) -
                          static_cast<float>(in_0_grad_data.at(i)) - 1);
    if (diff > 0.01) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff
                << " out0: " << static_cast<float>(host_in_0_grad_ptr[i])
                << " out1 " << static_cast<float>(in_0_grad_data.at(i)) + 1
                << std::endl;
    }
  }

  std::ifstream in_1_grad_is;
  in_1_grad_is.open("/data/guoran/data/backward_data/in_1_grad.bin");
  std::vector<half> in_1_grad_data(batch_size * feature_dims.at(1) *
                                   vector_size);
  in_1_grad_is.read(reinterpret_cast<char *>(in_1_grad_data.data()), in_1_size);

  for (int i = 0; i < batch_size * feature_dims.at(1) * vector_size; i++) {
    int batch_idx = i / (feature_dims.at(1) * vector_size);
    int out_i = i % (feature_dims.at(1) * vector_size);
    float diff = std::abs(static_cast<float>(host_in_1_grad_ptr[i]) -
                          static_cast<float>(in_1_grad_data.at(i)));
    if (diff > 0.01) {
      std::cout << "i " << i << " batch_idx" << batch_idx << " out_i " << out_i
                << " diff " << diff
                << " out0: " << static_cast<float>(host_in_1_grad_ptr[i])
                << " out1 " << static_cast<float>(in_1_grad_data.at(i))
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
