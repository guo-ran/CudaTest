#include <cuda_fp16.h>
#include <cudnn.h>
#include <fstream>
#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

void Check(cudnnStatus_t err) {
  if (err != CUDNN_STATUS_SUCCESS) {
    std::cerr << cudnnGetErrorString(err) << std::endl;
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

  cudnnHandle_t handle;
  Check(cudnnCreate(&handle));
  cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
  // cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
  cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NCHW;
  cudnnTensorDescriptor_t xDesc;
  Check(cudnnCreateTensorDescriptor(&xDesc));
  Check(cudnnSetTensor4dDescriptor(xDesc, cudnn_data_format, CUDNN_DATA_HALF,
                                   num_instances, norm_size, 1, 1));
  cudnnTensorDescriptor_t yDesc;
  Check(cudnnCreateTensorDescriptor(&yDesc));
  Check(cudnnSetTensor4dDescriptor(yDesc, cudnn_data_format, CUDNN_DATA_HALF,
                                   num_instances, norm_size, 1, 1));
  float one_val = 1.0f;
  float zero_val = 0.0f;
  const void *alpha = static_cast<const void *>(&one_val);
  const void *beta = static_cast<const void *>(&zero_val);
  Check(cudnnSoftmaxForward(handle, algorithm, mode, alpha, xDesc, x, beta,
                            yDesc, y));
  Check(cudaDeviceSynchronize());

  Check(cudnnDestroyTensorDescriptor(xDesc));
  Check(cudnnDestroyTensorDescriptor(yDesc));
  Check(cudnnDestroy(handle));
  Check(cudaFreeHost(in_host));
  Check(cudaFree(x));
  Check(cudaFree(y));
  return 0;
}
