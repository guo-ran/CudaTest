#include <fstream>
#include <iostream>
#include <thrust/pair.h>
#include <thrust/tuple.h>

constexpr int kCUDANumThreads = 256;
constexpr int kColwiseReduceTileSize = 32;
#define __ubsan_ignore_float_divide_by_zero__                                  \
  __attribute__((no_sanitize("float-divide-by-zero")))
#define C10_WARP_SIZE 32
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define device_sqrt std::sqrt

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

namespace cuda_utils {

constexpr int kCUDABlockReduceNumThreads = 512;

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp &op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T BlockReduce(T val, const ReduceOp &op,
                                    const T &identity_element, T *shared) {
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < blockDim.x / C10_WARP_SIZE) ? shared[lid]
                                                   : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

} // namespace cuda_utils

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  C10_HOST_DEVICE WelfordData(scalar_t mean, scalar_t m2, index_t n,
                              combine_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename scalar_t, typename acc_scalar_t, typename index_t,
          typename combine_t, typename res_t>
struct WelfordOps {
  index_t correction;
  bool take_sqrt;

public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data,
                                 index_t /*idx*/) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
        new_mean, acc.m2 + delta * new_delta, acc.n + 1,
        combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
        a.mean + delta * nb_over_n,
        a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
        // setting acc.n as -1 since acc.n might not be able to represent the
        // count correctly within its range, setting it to -1 to avoid confusion
        -1, new_count};
  }
  inline C10_DEVICE res_t
  project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const combine_t divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    res_t results(take_sqrt ? sqrtf(var) : var, mean);
    return results;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {WARP_SHFL_DOWN(acc.mean, offset), WARP_SHFL_DOWN(acc.m2, offset),
            WARP_SHFL_DOWN(acc.n, offset), WARP_SHFL_DOWN(acc.nf, offset)};
  }
#endif
  C10_HOST_DEVICE WelfordOps(index_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

template <typename T>
__global__ void RowwiseMomentsCUDAKernel(int64_t N, T eps, const T *X, T *mean,
                                         T *rstd) {
  using T_ACC = float;
  using WelfordType = WelfordData<T_ACC, int64_t, T_ACC>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, T_ACC, thrust::pair<T_ACC, T_ACC>>;

  __shared__ typename std::aligned_storage<sizeof(WelfordType),
                                           alignof(WelfordType)>::type
      val_shared[C10_WARP_SIZE];
  WelfordType *val_shared_ptr = reinterpret_cast<WelfordType *>(val_shared);

  const int64_t i = blockIdx.x;
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }
  val = cuda_utils::BlockReduce(val, welford_op,
                                /*identity_element=*/WelfordType(0, 0, 0, 0),
                                val_shared_ptr);

  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[i] = m1;
    rstd[i] = rsqrtf(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void LayerNormForwardCUDAKernel(int64_t N, const T *X, const T *mean,
                                           const T *rstd, const T *gamma,
                                           const T *beta, T *Y) {
  using T_ACC = float;
  const int64_t i = blockIdx.x;
  for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
    const int64_t index = i * N + j;
    const T_ACC gamma_v =
        gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
    const T_ACC beta_v =
        beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
    Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
                   static_cast<T_ACC>(rstd[i]) * gamma_v +
               beta_v;
  }
}

void forward(int num_instances, int norm_size) {
  std::cout << "num_instances " << num_instances << " norm_size " << norm_size
            << std::endl;
  int elem_cnt = num_instances * norm_size;
  float *x_ptr;
  float *y_ptr;
  float *mean_ptr;
  float *inv_variance_ptr;
  float *gamma_ptr;
  float *beta_ptr;
  double epsilon = 1e-5;
  cudaMalloc(&x_ptr, elem_cnt * sizeof(float));
  cudaMalloc(&y_ptr, elem_cnt * sizeof(float));
  cudaMalloc(&mean_ptr, num_instances * sizeof(float));
  cudaMalloc(&inv_variance_ptr, num_instances * sizeof(float));
  cudaMalloc(&gamma_ptr, norm_size * sizeof(float));
  cudaMalloc(&beta_ptr, norm_size * sizeof(float));
  float *x_host;
  cudaMallocHost(&x_host, elem_cnt * sizeof(float));
  std::ifstream x_is;
  x_is.open("data.bin");
  x_is.read(reinterpret_cast<char *>(x_host), elem_cnt * sizeof(float));
  x_is.close();
  printf("in 100 %f", static_cast<float>(x_host[100]));
  cudaMemcpy(x_ptr, x_host, elem_cnt * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(gamma_ptr, x_host, norm_size * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(beta_ptr, x_host, norm_size * sizeof(float), cudaMemcpyDefault);
  int M = num_instances;
  int N = norm_size;
  RowwiseMomentsCUDAKernel<float>
      <<<M, cuda_utils::kCUDABlockReduceNumThreads, 0, nullptr>>>(
          N, epsilon, x_ptr, mean_ptr, inv_variance_ptr);

  LayerNormForwardCUDAKernel<float><<<M, kCUDANumThreads, 0, nullptr>>>(
      N, x_ptr, mean_ptr, inv_variance_ptr, gamma_ptr, beta_ptr, y_ptr);

  cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
  int num_instances = atoi(argv[1]);
  int norm_size = atoi(argv[2]);
  forward(num_instances, norm_size);
  return 0;
}
