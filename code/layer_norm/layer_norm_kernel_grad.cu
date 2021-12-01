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

template <typename SRC, typename DST, bool do_scale> struct ScaleLoad {
  ScaleLoad(const SRC *src, const SRC *gamma, int64_t row_size)
      : src(src), gamma(gamma), row_size(row_size) {}
  template <int N>
  __device__ void load(DST *dst, int64_t row, int64_t col) const {
    cuda::layer_norm::Pack<SRC, N> src_pack;
    cuda::layer_norm::Pack<SRC, N> gamma_pack;
    const int64_t offset = row * row_size + col;
    src_pack.storage =
        *reinterpret_cast<const cuda::layer_norm::PackType<SRC, N> *>(src +
                                                                      offset);
    if (do_scale) {
      gamma_pack.storage =
          *reinterpret_cast<const cuda::layer_norm::PackType<SRC, N> *>(gamma +
                                                                        col);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        gamma_pack.elem[i] = 1;
      }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(src_pack.elem[i] * gamma_pack.elem[i]);
    }
  }
  const SRC *src;
  const SRC *gamma;
  int64_t row_size;
};

} // namespace oneflow

int main(int argc, char **argv) {
  int num_instances = atoi(argv[1]);
  int norm_size = atoi(argv[2]);
  std::cout << "num_instances " << num_instances << " norm_size " << norm_size
            << std::endl;
  int elem_cnt = num_instances * norm_size;
  half *x_ptr;
  half *dy_ptr;
  half *dx_ptr;
  float *mean_ptr;
  float *inv_variance_ptr;
  half *gamma_ptr;
  half *in_host;
  Check(cudaMallocHost(&in_host, elem_cnt * sizeof(half)));
  std::ifstream x_is;
  x_is.open("data.bin");
  x_is.read(reinterpret_cast<char *>(in_host), elem_cnt * sizeof(half));
  x_is.close();
  printf("in 100 %f", static_cast<float>(in_host[100]));
  Check(cudaMalloc(&x_ptr, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&dy_ptr, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&dx_ptr, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&mean_ptr, num_instances * sizeof(float)));
  Check(cudaMalloc(&inv_variance_ptr, num_instances * sizeof(float)));
  Check(cudaMalloc(&gamma_ptr, norm_size * sizeof(half)));

  Check(cudaMemcpy(x_ptr, in_host, elem_cnt * sizeof(half), cudaMemcpyDefault));
  Check(
      cudaMemcpy(dy_ptr, in_host, elem_cnt * sizeof(half), cudaMemcpyDefault));
  Check(cudaMemcpy(gamma_ptr, in_host, norm_size * sizeof(half),
                   cudaMemcpyDefault));
  Check(cudaMemcpy(mean_ptr, in_host, num_instances * sizeof(half),
                   cudaMemcpyDefault));
  Check(cudaMemcpy(inv_variance_ptr, in_host, num_instances * sizeof(half),
                   cudaMemcpyDefault));

  oneflow::cuda::layer_norm::DirectLoad<half, float> load_x(x_ptr, norm_size);
  oneflow::ScaleLoad<half, float, true> load_dy(dy_ptr, gamma_ptr, norm_size);
  oneflow::cuda::layer_norm::DirectStore<float, half> store(dx_ptr, norm_size);
  Check(oneflow::cuda::layer_norm::DispatchLayerNormGrad<
        decltype(load_x), decltype(load_dy), decltype(store), float>(
      nullptr, load_x, load_dy, store, mean_ptr, inv_variance_ptr,
      num_instances, norm_size));

  Check(cudaDeviceSynchronize());
  return 0;
}
