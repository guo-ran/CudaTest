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
      if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        y_pack.elem[i] = normalized_i;
      }
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

int main(int argc, char **argv) {
  int num_instances = atoi(argv[1]);
  int norm_size = atoi(argv[2]);
  std::cout << "num_instances " << num_instances << " norm_size " << norm_size
            << std::endl;
  int elem_cnt = num_instances * norm_size;
  half *x_ptr;
  half *y_ptr;
  float *mean_ptr;
  float *inv_variance_ptr;
  half *gamma_ptr;
  half *beta_ptr;
  double epsilon = 1e-5;
  half *in_host;
  Check(cudaMallocHost(&in_host, elem_cnt * sizeof(half)));
  std::ifstream x_is;
  x_is.open("data.bin");
  x_is.read(reinterpret_cast<char *>(in_host), elem_cnt * sizeof(half));
  x_is.close();
  printf("in 100 %f", static_cast<float>(in_host[100]));
  Check(cudaMalloc(&x_ptr, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&y_ptr, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&mean_ptr, num_instances * sizeof(float)));
  Check(cudaMalloc(&inv_variance_ptr, num_instances * sizeof(float)));
  Check(cudaMalloc(&gamma_ptr, norm_size * sizeof(half)));
  Check(cudaMalloc(&beta_ptr, norm_size * sizeof(half)));
  Check(cudaMemcpy(x_ptr, in_host, elem_cnt * sizeof(half), cudaMemcpyDefault));
  Check(cudaMemcpy(gamma_ptr, in_host, norm_size * sizeof(half),
                   cudaMemcpyDefault));
  Check(cudaMemcpy(beta_ptr, in_host, norm_size * sizeof(half),
                   cudaMemcpyDefault));
  oneflow::cuda::layer_norm::DirectLoad<half, float> load(x_ptr, norm_size);
  oneflow::AffineStore<float, half, true, true> store(y_ptr, norm_size,
                                                      gamma_ptr, beta_ptr);
  Check(oneflow::cuda::layer_norm::DispatchLayerNorm<decltype(load),
                                                     decltype(store), float>(
      nullptr, load, store, num_instances, norm_size, epsilon, mean_ptr,
      inv_variance_ptr));
  Check(cudaDeviceSynchronize());
  return 0;
}
