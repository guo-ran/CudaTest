#include "layer_norm.cuh"
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}
namespace oneflow {

template <typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(DST *y, int64_t row_size, const DST *gamma, const DST *beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template <int N>
  __device__ void store(const SRC *src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> y_pack;
    cuda::layer_norm::Pack<DST, N> gamma_pack;
    cuda::layer_norm::Pack<DST, N> beta_pack;
    const int64_t offset = row * row_size + col;
    if (do_scale) {
      gamma_pack.storage =
          *reinterpret_cast<const cuda::layer_norm::PackType<DST, N> *>(gamma +
                                                                        col);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        gamma_pack.elem[i] = 1;
      }
    }
    if (do_center) {
      beta_pack.storage =
          *reinterpret_cast<const cuda::layer_norm::PackType<DST, N> *>(beta +
                                                                        col);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        beta_pack.elem[i] = 0;
      }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      //if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      //} else {
      //  y_pack.elem[i] = normalized_i;
      //}
    }
    *reinterpret_cast<cuda::layer_norm::PackType<DST, N> *>(y + offset) =
        y_pack.storage;
  }
  DST *y;
  int64_t row_size;
  const DST *gamma;
  const DST *beta;
};

} // namespace oneflow

template <typename T> void forward(int num_instances, int norm_size) {
  using ComputeType =
      typename oneflow::cuda::layer_norm::DefaultComputeType<T>::type;
  int elem_cnt = num_instances * norm_size;
  T *x_ptr;
  T *y_ptr;
  ComputeType *mean_ptr;
  ComputeType *inv_variance_ptr;
  T *gamma_ptr;
  T *beta_ptr;
  double epsilon = 1e-5;
  T *in_host;
  Check(cudaMallocHost(&in_host, elem_cnt * sizeof(T)));
  std::ifstream x_is;
  x_is.open("data.bin");
  x_is.read(reinterpret_cast<char *>(in_host), elem_cnt * sizeof(half));
  x_is.close();
  printf("in 100 %f", static_cast<float>(in_host[100]));
  Check(cudaMalloc(&x_ptr, elem_cnt * sizeof(T)));
  Check(cudaMalloc(&y_ptr, elem_cnt * sizeof(T)));
  Check(cudaMalloc(&mean_ptr, num_instances * sizeof(ComputeType)));
  Check(cudaMalloc(&inv_variance_ptr, num_instances * sizeof(ComputeType)));
  Check(cudaMalloc(&gamma_ptr, norm_size * sizeof(T)));
  Check(cudaMalloc(&beta_ptr, norm_size * sizeof(T)));
  Check(cudaMemcpy(x_ptr, in_host, elem_cnt * sizeof(T), cudaMemcpyDefault));
  Check(
      cudaMemcpy(gamma_ptr, in_host, norm_size * sizeof(T), cudaMemcpyDefault));
  Check(
      cudaMemcpy(beta_ptr, in_host, norm_size * sizeof(T), cudaMemcpyDefault));
  oneflow::cuda::layer_norm::DirectLoad<T, ComputeType> load(x_ptr, norm_size);
  oneflow::AffineStore<ComputeType, T, true, true> store(y_ptr, norm_size,
                                                         gamma_ptr, beta_ptr);
  Check(oneflow::cuda::layer_norm::DispatchLayerNorm<
        decltype(load), decltype(store), ComputeType>(
      nullptr, load, store, num_instances, norm_size, epsilon, mean_ptr,
      inv_variance_ptr));
  Check(cudaDeviceSynchronize());
  Check(cudaFreeHost(in_host));
  Check(cudaFree(x_ptr));
  Check(cudaFree(y_ptr));
  Check(cudaFree(mean_ptr));
  Check(cudaFree(inv_variance_ptr));
  Check(cudaFree(gamma_ptr));
  Check(cudaFree(beta_ptr));
}

int main(int argc, char **argv) {
  int num_instances = atoi(argv[1]);
  int norm_size = atoi(argv[2]);
  std::cout << "num_instances " << num_instances << " norm_size " << norm_size
            << std::endl;
  forward<half>(num_instances, norm_size);
  // forward<float>(num_instances, norm_size);

  return 0;
}
