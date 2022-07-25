#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

void CudaCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

constexpr int kWarpSize = 32;

template <typename T> struct SumOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a + b;
  }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T WarpReduce(T val) {
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_down_sync(0xffffffff, val, mask));
  }
  return val;
}

constexpr int tile_size = 32;
constexpr int num_per_block = 4;
constexpr int block_dim_x = 32;
constexpr int block_dim_y = 32 / num_per_block;

template <typename T, typename ComputeType>
__global__ void FeatureInteractionSumSharedMemImpl(int64_t batch_size,
                                                   int64_t feature_dim,
                                                   int64_t embedding_size,
                                                   const T *in, T *out) {
  __shared__ ComputeType in_buf[32][33];
  __shared__ ComputeType in_square_buf[32][33];
  for (int batch_idx = blockIdx.y; batch_idx < batch_size;
       batch_idx += gridDim.y) {
    const T *batch_in = in + batch_idx * feature_dim * embedding_size;
    T *batch_out = out + batch_idx * embedding_size;
    int col_id = blockIdx.x * 32 + threadIdx.x;
    int buf_col_id = threadIdx.x;
    for (int row = threadIdx.y; row < feature_dim;
         row += blockDim.y * num_per_block) {
      for (int i = 0; i < num_per_block; ++i) {
        int row_id = row + i * blockDim.y;
        int buf_row_id = threadIdx.y + i * blockDim.y;
        if (row_id < 32) {
          in_buf[buf_row_id][buf_col_id] = 0;
          in_square_buf[buf_row_id][buf_col_id] = 0;
        }
        if (col_id < embedding_size && row_id < feature_dim) {
          const ComputeType in_val = static_cast<ComputeType>(
              batch_in[row_id * embedding_size + col_id]);
          in_buf[buf_row_id][buf_col_id] += in_val;
          in_square_buf[buf_row_id][buf_col_id] += in_val * in_val;
        }
      }
    }
    __syncwarp();
    for (int i = 0; i < num_per_block; ++i) {
      int buf_col_id = threadIdx.y + i * blockDim.y;
      ComputeType thread_val = in_buf[threadIdx.x][buf_col_id];
      ComputeType thread_square_val = in_square_buf[threadIdx.x][buf_col_id];
      ComputeType sum = WarpReduce<SumOp, ComputeType>(thread_val);
      ComputeType square_of_sum =
          WarpReduce<SumOp, ComputeType>(thread_square_val);
      if (threadIdx.x == 0) {
        ComputeType bi_interaction =
            (sum * sum - square_of_sum) * static_cast<ComputeType>(0.5);
        batch_out[blockIdx.x * 32 + buf_col_id] =
            static_cast<T>(bi_interaction);
      }
    }
  }
}

template <typename T>
__global__ void FeatureInteractionSumSharedMemImpl2(int batch_size,
                                                    int feature_dim,
                                                    const int embedding_size,
                                                    const T *in, T *out) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  int warp_id = threadIdx.y;
  T *buf = reinterpret_cast<T *>(shared_buf) + warp_id * embedding_size;
  T *square_buf = reinterpret_cast<T *>(shared_buf) +
                  blockDim.y * embedding_size + warp_id * embedding_size;
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
       batch_idx < batch_size; batch_idx += gridDim.x * blockDim.y) {
    const T *batch_in = in + batch_idx * feature_dim * embedding_size;
    T *batch_out = out + batch_idx * embedding_size;
    for (int thread_id = threadIdx.x; thread_id < feature_dim * embedding_size;
         thread_id += blockDim.x) {
      int row_id = thread_id / embedding_size;
      int col_id = thread_id - row_id * embedding_size;
      const T val = batch_in[row_id * embedding_size + col_id];
      // TODO: buf not initialized
      atomicAdd(buf + col_id, val);
      atomicAdd(square_buf + col_id, val * val);
    }
    __syncwarp();
    for (int i = threadIdx.x; i < embedding_size; i += blockDim.x) {
      const T sum = buf[i];
      const T square_sum = square_buf[i];
      batch_out[i] = (sum * sum - square_sum) * static_cast<T>(0.5);
    }
  }
}

// hugectr impl
__global__ void fm_order2_kernel(const __half *in, __half *out, int batch_size,
                                 int slot_num, int emb_vec_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid < emb_vec_size && bid < batch_size) {
    __half emb_sum = __float2half(0.0f);
    __half emb_sum_square = __float2half(0.0f);
    __half emb_square_sum = __float2half(0.0f);
    int offset = bid * slot_num * emb_vec_size + tid;

    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      __half temp = in[index];
      emb_sum += temp;
      emb_square_sum += temp * temp;
    }
    emb_sum_square = emb_sum * emb_sum;

    out[bid * emb_vec_size + tid] =
        __float2half(0.5f) * (emb_sum_square - emb_square_sum);
  }
}

__global__ void fm_order2_dgrad_kernel(const float* in, const float* top_grad, float* dgrad,
                                       int batch_size, int slot_num, int emb_vec_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid < emb_vec_size && bid < batch_size) {
    float emb_sum = 0.0f;
    int offset = bid * slot_num * emb_vec_size + tid;

    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      emb_sum += in[index];
    }
    float tgrad = top_grad[bid * emb_vec_size + tid];
    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      dgrad[index] = tgrad * (emb_sum - in[index]);
    }
  }
}

// hugectr impl
__global__ void fm_order2_dgrad_kernel(const __half *in, const __half *top_grad,
                                       __half *dgrad, int batch_size,
                                       int slot_num, int emb_vec_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (tid < emb_vec_size && bid < batch_size) {
    __half emb_sum = __float2half(0.0f);
    int offset = bid * slot_num * emb_vec_size + tid;

    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      emb_sum += in[index];
    }
    __half tgrad = top_grad[bid * emb_vec_size + tid];
    for (int i = 0; i < slot_num; i++) {
      int index = offset + i * emb_vec_size;
      dgrad[index] = tgrad * (emb_sum - in[index]);
    }
  }
}

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

constexpr int pack_size = 2;

template <typename T>
__global__ void FillValue(int64_t elem_cnt, T value, T *ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    ptr[i] = value;
  }
}

template <typename T> struct DefaultComputeType { using type = T; };

template <> struct DefaultComputeType<half> { using type = float; };

template <typename T, int32_t N> struct Param {
  const T *in[N];
  int32_t in_feature_dim[N];
  T *out;
  int32_t num_in;
};

template <typename T, int32_t N>
__global__ void FeatureInteractionSum(int64_t batch_size,
                                      int64_t embedding_size,
                                      Param<T, N> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
       batch_idx < batch_size; batch_idx += gridDim.x * blockDim.y) {
    T *batch_out = param.out + batch_idx * embedding_size;
    for (int col_id = threadIdx.x; col_id < embedding_size;
         col_id += blockDim.x) {
      ComputeType sum = 0;
      ComputeType square_sum = 0;
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) {
          break;
        }
        const T *batch_in =
            param.in[i] + batch_idx * param.in_feature_dim[i] * embedding_size;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          ComputeType val =
              static_cast<ComputeType>(batch_in[j * embedding_size + col_id]);
          sum += val;
          square_sum += val * val;
        }
      }
      batch_out[col_id] = static_cast<T>((sum * sum - square_sum) *
                                         static_cast<ComputeType>(0.5));
    }
  }
}

// embedding_size must / pack_size
template <typename T, int32_t N>
__global__ void FeatureInteractionSumPack(int64_t batch_size,
                                          int64_t embedding_size,
                                          Param<T, N> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  Pack<T, pack_size> *dst_pack =
      reinterpret_cast<Pack<T, pack_size> *>(param.out);
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
       batch_idx < batch_size; batch_idx += gridDim.x * blockDim.y) {
    Pack<T, pack_size> *batch_out = dst_pack + batch_idx * embedding_size;
    for (int col_id = threadIdx.x; col_id < embedding_size;
         col_id += blockDim.x) {
      Pack<ComputeType, pack_size> sum;
      Pack<ComputeType, pack_size> square_sum;
#pragma unroll
      for (int k = 0; k < pack_size; ++k) {
        sum.elem[k] = static_cast<ComputeType>(0);
        square_sum.elem[k] = static_cast<ComputeType>(0);
      }
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) {
          break;
        }
        const Pack<T, pack_size> *batch_in =
            reinterpret_cast<const Pack<T, pack_size> *>(param.in[i]) +
            batch_idx * param.in_feature_dim[i] * embedding_size;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          Pack<T, pack_size> val = batch_in[j * embedding_size + col_id];
#pragma unroll
          for (int k = 0; k < pack_size; ++k) {
            const ComputeType compute_val =
                static_cast<ComputeType>(val.elem[k]);
            sum.elem[k] += compute_val;
            square_sum.elem[k] += compute_val * compute_val;
          }
        }
      }
      Pack<T, pack_size> out;
#pragma unroll
      for (int k = 0; k < pack_size; ++k) {
        out.elem[k] =
            static_cast<T>((sum.elem[k] * sum.elem[k] - square_sum.elem[k]) *
                           static_cast<ComputeType>(0.5));
      }
      batch_out[col_id] = out;
    }
  }
}

template <typename T, int32_t N> struct GradParam {
  const T *dy;
  const T *in[N];
  int32_t in_feature_dim[N];
  T *in_grad[N];
  int32_t num_in;
};

template <typename T, int32_t N>
__global__ void FeatureInteractionSumGrad(int64_t batch_size,
                                          int64_t embedding_size,
                                          GradParam<T, N> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
       batch_idx < batch_size; batch_idx += gridDim.x * blockDim.y) {
    const T *batch_dy = param.dy + batch_idx * embedding_size;
    for (int col_id = threadIdx.x; col_id < embedding_size;
         col_id += blockDim.x) {
      ComputeType sum = 0;
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) {
          break;
        }
        const T *batch_in =
            param.in[i] + batch_idx * param.in_feature_dim[i] * embedding_size;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          sum +=
              static_cast<ComputeType>(batch_in[j * embedding_size + col_id]);
        }
      }
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) {
          break;
        }
        const int64_t in_batch_offset =
            batch_idx * param.in_feature_dim[i] * embedding_size;
        const T *batch_in = param.in[i] + in_batch_offset;
        T *batch_in_grad = param.in_grad[i] + in_batch_offset;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          const int64_t offset = j * embedding_size + col_id;
          batch_in_grad[offset] = static_cast<T>(
              static_cast<ComputeType>(batch_dy[col_id]) *
              (sum - static_cast<ComputeType>(batch_in[offset])));
        }
      }
    }
  }
}

template <typename T, int32_t N>
__global__ void FeatureInteractionSumGradPack(int64_t batch_size,
                                              int64_t embedding_size,
                                              GradParam<T, N> param) {
  using ComputeType = typename DefaultComputeType<T>::type;
  const Pack<T, pack_size> *dy_pack =
      reinterpret_cast<const Pack<T, pack_size> *>(param.dy);
  for (int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
       batch_idx < batch_size; batch_idx += gridDim.x * blockDim.y) {
    const Pack<T, pack_size> *batch_dy = dy_pack + batch_idx * embedding_size;
    for (int col_id = threadIdx.x; col_id < embedding_size;
         col_id += blockDim.x) {
      Pack<ComputeType, pack_size> sum;
#pragma unroll
      for (int k = 0; k < pack_size; ++k) {
        sum.elem[k] = static_cast<ComputeType>(0);
      }
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) {
          break;
        }
        const Pack<T, pack_size> *batch_in =
            reinterpret_cast<const Pack<T, pack_size> *>(param.in[i]) +
            batch_idx * param.in_feature_dim[i] * embedding_size;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          Pack<T, pack_size> val = batch_in[j * embedding_size + col_id];
#pragma unroll
          for (int k = 0; k < pack_size; ++k) {
            const ComputeType compute_val =
                static_cast<ComputeType>(val.elem[k]);
            sum.elem[k] += compute_val;
          }
        }
      }
      for (int i = 0; i < N; ++i) {
        if (i >= param.num_in) {
          break;
        }
        const int64_t in_batch_offset =
            batch_idx * param.in_feature_dim[i] * embedding_size;
        const Pack<T, pack_size> *batch_in =
            reinterpret_cast<const Pack<T, pack_size> *>(param.in[i]) +
            in_batch_offset;
        Pack<T, pack_size> *batch_in_grad =
            reinterpret_cast<Pack<T, pack_size> *>(param.in_grad[i]) +
            in_batch_offset;
        for (int j = 0; j < param.in_feature_dim[i]; ++j) {
          const int64_t offset = j * embedding_size + col_id;
          Pack<T, pack_size> in_grad_val;
          const Pack<T, pack_size> dy_val = batch_dy[col_id];
          const Pack<T, pack_size> in_val = batch_in[offset];
#pragma unroll
          for (int k = 0; k < pack_size; ++k) {
            in_grad_val.elem[k] = static_cast<T>(
                static_cast<ComputeType>(dy_val.elem[k]) *
                (sum.elem[k] - static_cast<ComputeType>(in_val.elem[k])));
          }
          batch_in_grad[offset] = in_grad_val;
        }
      }
    }
  }
}

void GetBlockDims(const int64_t vector_size, int *block_dim_x,
                  int *block_dim_y) {
  const int block_size = 256;
  if (vector_size < block_size) {
    *block_dim_x = vector_size;
    *block_dim_y = (block_size + vector_size - 1) / vector_size;
  } else {
    *block_dim_x = block_size;
    *block_dim_y = 1;
  }
}

constexpr int kCudaMaxBlocksNum = 8192;

int GetNumBlocks(const int64_t num_instances,
                 const int64_t instance_per_block) {
  int max_blocks =
      (num_instances + instance_per_block - 1) / instance_per_block;
  return std::min(max_blocks, kCudaMaxBlocksNum);
}

int main() {
  using T = half; // int
  int64_t batch_size = 55296 / 8;
  int64_t vector_size = 16;
  int64_t feature_dim = 39;

  int64_t elem_cnt = batch_size * vector_size * feature_dim;
  T *in_ptr;
  T *in_grad_ptr;
  T *out_ptr;
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));
  CudaCheck(cudaMalloc(&in_ptr, elem_cnt * sizeof(T)));
  CudaCheck(cudaMalloc(&in_grad_ptr, elem_cnt * sizeof(T)));
  CudaCheck(cudaMalloc(&out_ptr, elem_cnt * sizeof(T)));
  FillValue<<<elem_cnt / 1024, 1024, 0, stream>>>(elem_cnt, static_cast<T>(4),
                                                  in_ptr);

  { // forward hugectr
    dim3 blockSize(vector_size, 1, 1);
    dim3 grdiSize(batch_size, 1, 1);
    fm_order2_kernel<<<grdiSize, blockSize, 0, stream>>>(
        in_ptr, out_ptr, batch_size, feature_dim, vector_size);
  }
  { // forward pack
    int block_dim_x;
    int block_dim_y;
    int vector_num_pack = vector_size / pack_size;
    GetBlockDims(vector_num_pack, &block_dim_x, &block_dim_y);
    const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
    dim3 block_dims = dim3(block_dim_x, block_dim_y);
    Param<T, 1> param;
    param.in[0] = in_ptr;
    param.in_feature_dim[0] = feature_dim;
    param.num_in = 1;
    param.out = out_ptr;
    FeatureInteractionSumPack<T, 1><<<num_blocks, block_dims, 0, stream>>>(
        batch_size, vector_num_pack, param);
  }
  { // forward
    int block_dim_x;
    int block_dim_y;
    GetBlockDims(vector_size, &block_dim_x, &block_dim_y);
    const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
    dim3 block_dims = dim3(block_dim_x, block_dim_y);
    Param<T, 1> param;
    param.in[0] = in_ptr;
    param.in_feature_dim[0] = feature_dim;
    param.num_in = 1;
    param.out = out_ptr;
    FeatureInteractionSum<T, 1>
        <<<num_blocks, block_dims, 0, stream>>>(batch_size, vector_size, param);
  }
  { // backward hugectr
    dim3 blockSize(vector_size, 1, 1);
    dim3 gridSize(batch_size, 1, 1);
    fm_order2_dgrad_kernel<<<gridSize, blockSize, 0, stream>>>(
        in_ptr, out_ptr, in_grad_ptr, batch_size, feature_dim, vector_size);
  }
  { // backward pack
    int block_dim_x;
    int block_dim_y;
    int vector_num_pack = vector_size / pack_size;
    GetBlockDims(vector_num_pack, &block_dim_x, &block_dim_y);
    const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
    dim3 block_dims = dim3(block_dim_x, block_dim_y);
    GradParam<T, 1> param;
    param.in[0] = in_ptr;
    param.in_grad[0] = in_grad_ptr;
    param.in_feature_dim[0] = feature_dim;
    param.num_in = 1;
    param.dy = out_ptr;
    FeatureInteractionSumGradPack<T, 1><<<num_blocks, block_dims, 0, stream>>>(
        batch_size, vector_num_pack, param);
  }
  { // backward
    int block_dim_x;
    int block_dim_y;
    GetBlockDims(vector_size, &block_dim_x, &block_dim_y);
    const int num_blocks = GetNumBlocks(batch_size, block_dim_y);
    dim3 block_dims = dim3(block_dim_x, block_dim_y);
    GradParam<T, 1> param;
    param.in[0] = in_ptr;
    param.in_grad[0] = in_grad_ptr;
    param.in_feature_dim[0] = feature_dim;
    param.num_in = 1;
    param.dy = out_ptr;
    FeatureInteractionSumGrad<T, 1>
        <<<num_blocks, block_dims, 0, stream>>>(batch_size, vector_size, param);
  }
  CudaCheck(cudaStreamSynchronize(stream));
  CudaCheck(cudaFree(in_ptr));
  CudaCheck(cudaFree(out_ptr));
  return 0;
}
