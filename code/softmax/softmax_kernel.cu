#include "softmax.cuh"
#include <algorithm>
#include <fstream>
#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}
int main(int argc, char **argv) {
  int num_instances = atoi(argv[1]);
  int norm_size = atoi(argv[2]);
  std::cout << "num_instances " << num_instances << " norm_size " << norm_size
            << std::endl;
  int elem_cnt = num_instances * norm_size;
  half *x;
  half *y;

  Check(cudaMalloc(&x, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&y, elem_cnt * sizeof(half)));

  half *in_host;
  Check(cudaMallocHost(&in_host, elem_cnt * sizeof(half)));
  std::ifstream x_is;
  x_is.open("./data.bin");
  x_is.read(reinterpret_cast<char *>(in_host), elem_cnt * sizeof(half));
  x_is.close();
  std::cout << "in " << static_cast<float>(in_host[2]) << std::endl;
  Check(cudaMemcpy(x, in_host, elem_cnt * sizeof(half), cudaMemcpyDefault));

  oneflow::cuda::softmax::DirectLoad<half, float> load(x, norm_size);
  oneflow::cuda::softmax::DirectStore<float, half> store(y, norm_size);
  Check(oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store),
                                                float>(
      nullptr, load, store, num_instances, norm_size));

  Check(cudaFreeHost(in_host));
  Check(cudaFree(x));
  Check(cudaFree(y));

  Check(cudaDeviceSynchronize());
  return 0;
}
