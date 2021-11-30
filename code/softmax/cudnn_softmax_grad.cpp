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

  cudnnHandle_t handle;
  Check(cudnnCreate(&handle));
  cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
  // cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;
  cudnnTensorFormat_t cudnn_data_format = CUDNN_TENSOR_NCHW;
  cudnnTensorDescriptor_t yDesc;
  Check(cudnnCreateTensorDescriptor(&yDesc));
  Check(cudnnSetTensor4dDescriptor(yDesc, cudnn_data_format, CUDNN_DATA_HALF,
                                   num_instances, norm_size, 1, 1));
  cudnnTensorDescriptor_t dyDesc;
  Check(cudnnCreateTensorDescriptor(&dyDesc));
  Check(cudnnSetTensor4dDescriptor(dyDesc, cudnn_data_format, CUDNN_DATA_HALF,
                                   num_instances, norm_size, 1, 1));
  cudnnTensorDescriptor_t dxDesc;
  Check(cudnnCreateTensorDescriptor(&dxDesc));
  Check(cudnnSetTensor4dDescriptor(dxDesc, cudnn_data_format, CUDNN_DATA_HALF,
                                   num_instances, norm_size, 1, 1));
  float one_val = 1.0f;
  float zero_val = 0.0f;
  const void *alpha = static_cast<const void *>(&one_val);
  const void *beta = static_cast<const void *>(&zero_val);
  Check(cudnnSoftmaxBackward(handle, algorithm, mode, alpha, yDesc, y, dyDesc,
                             dy, beta, dxDesc, dx));
  Check(cudaDeviceSynchronize());

  Check(cudnnDestroyTensorDescriptor(yDesc));
  Check(cudnnDestroyTensorDescriptor(dyDesc));
  Check(cudnnDestroyTensorDescriptor(dxDesc));
  Check(cudnnDestroy(handle));
  Check(cudaFreeHost(in_host));
  Check(cudaFree(y));
  Check(cudaFree(dy));
  Check(cudaFree(dx));
  return 0;
}
