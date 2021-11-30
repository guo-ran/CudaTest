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
  half *y;
  half *dy;
  half *dx;
  Check(cudaMalloc(&y, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&dy, elem_cnt * sizeof(half)));
  Check(cudaMalloc(&dx, elem_cnt * sizeof(half)));

  half *in_host;
  Check(cudaMallocHost(&in_host, elem_cnt * sizeof(half)));
  std::ifstream x_is;
  x_is.open("./data.bin");
  x_is.read(reinterpret_cast<char *>(in_host), elem_cnt * sizeof(half));
  x_is.close();
  Check(cudaMemcpy(dy, in_host, elem_cnt * sizeof(half), cudaMemcpyDefault));
  Check(cudaMemcpy(y, in_host, elem_cnt * sizeof(half), cudaMemcpyDefault));

  oneflow::cuda::softmax::DirectLoad<half, float> load_y(y, norm_size);
  oneflow::cuda::softmax::DirectLoad<half, float> load_dy(dy, norm_size);
  oneflow::cuda::softmax::DirectStore<float, half> store(dx, norm_size);
  Check(oneflow::cuda::softmax::DispatchSoftmaxGrad<
        decltype(load_y), decltype(load_dy), decltype(store), float>(
      nullptr, load_y, load_dy, store, num_instances, norm_size));

  cudaDeviceSynchronize();
  Check(cudaFreeHost(in_host));
  Check(cudaFree(y));
  Check(cudaFree(dy));
  Check(cudaFree(dx));
  return 0;
}
