#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void CudaCheck(cudnnStatus_t statue) {
  if (statue != CUDNN_STATUS_SUCCESS) {
    std::cerr << "cudnn error: " << cudnnGetErrorString(statue) << std::endl;
  }
}

void CudaCheck(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "cuda error: " << cudaGetErrorString(error) << std::endl;
  }
}

template <typename T> cudnnDataType_t GetCudnnDataType();

template <> cudnnDataType_t GetCudnnDataType<float>() {
  return cudnnDataType_t::CUDNN_DATA_FLOAT;
}

template <> cudnnDataType_t GetCudnnDataType<double>() {
  return cudnnDataType_t::CUDNN_DATA_DOUBLE;
}

template <> cudnnDataType_t GetCudnnDataType<half>() {
  return cudnnDataType_t::CUDNN_DATA_HALF;
}

#if CUDA_VERSION >= 11000
template <> cudnnDataType_t GetCudnnDataType<nv_bfloat16>() {
  return cudnnDataType_t::CUDNN_DATA_BFLOAT16;
}
#endif

template <typename T> cudnnDataType_t GetConvDescDataType() {
  return cudnnDataType_t::CUDNN_DATA_FLOAT;
}

template <> cudnnDataType_t GetConvDescDataType<double>() {
  return cudnnDataType_t::CUDNN_DATA_DOUBLE;
}

} // namespace

template <typename T>
void ConvForwardSearchAlgo(bool heuristic_search, cudnnHandle_t handle,
                           cudnnTensorDescriptor_t x_desc,
                           cudnnTensorDescriptor_t y_desc,
                           cudnnFilterDescriptor_t w_desc,
                           cudnnConvolutionDescriptor_t conv_desc, T *x_ptr,
                           T *w_ptr, T *y_ptr, void *workspace,
                           size_t workspace_size) {
  int count;
  cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &count);
  std::vector<cudnnConvolutionFwdAlgoPerf_t> res;
  res.resize(count);
  int ret;
  if (heuristic_search) {
    cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc, w_desc, conv_desc,
                                           y_desc, count, &ret, res.data());
  } else {
    cudnnFindConvolutionForwardAlgorithmEx(
        handle, x_desc, x_ptr, w_desc, w_ptr, conv_desc, y_desc, y_ptr, count,
        &ret, res.data(), workspace, workspace_size);
  }
  std::cout << "search forward algorithm:" << std::endl;
  for (int i = 0; i < ret; ++i) {
    std::cout << "algo: " << res.at(i).algo
              << " mathtype: " << res.at(i).mathType
              << " time: " << res.at(i).time << " memory: " << res.at(i).memory
              << " " << std::endl;
  }
}

template <typename T>
void ConvForward(int algo, cudnnHandle_t handle, cudnnTensorDescriptor_t x_desc,
                 cudnnTensorDescriptor_t y_desc, cudnnFilterDescriptor_t w_desc,
                 cudnnConvolutionDescriptor_t conv_desc, T *x_ptr, T *w_ptr,
                 T *y_ptr, void *workspace, size_t workspace_size) {
  float zero = 0.0;
  float one = 1.0;
  CudaCheck(cudnnConvolutionForward(
      handle, &one, x_desc, x_ptr, w_desc, w_ptr, conv_desc,
      static_cast<cudnnConvolutionFwdAlgo_t>(algo), workspace, workspace_size,
      &zero, y_desc, y_ptr));
}

template <typename T>
void ConvFilterBackwardSearchAlgo(bool heuristic_search, cudnnHandle_t handle,
                                  cudnnTensorDescriptor_t x_desc,
                                  cudnnTensorDescriptor_t y_desc,
                                  cudnnFilterDescriptor_t w_desc,
                                  cudnnConvolutionDescriptor_t conv_desc,
                                  T *x_ptr, T *w_ptr, T *y_ptr, void *workspace,
                                  size_t workspace_size) {
  int count;
  cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &count);
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> res;
  res.resize(count);
  int ret;
  if (heuristic_search) {
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, x_desc, y_desc, conv_desc, w_desc, count, &ret, res.data());
  } else {
    cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle, x_desc, x_ptr, y_desc, y_ptr, conv_desc, w_desc, w_ptr, count,
        &ret, res.data(), workspace, workspace_size);
  }
  std::cout << "search backward filter algorithm:" << std::endl;
  for (int i = 0; i < ret; ++i) {
    std::cout << "algo: " << res.at(i).algo
              << " mathtype: " << res.at(i).mathType
              << " time: " << res.at(i).time << " memory: " << res.at(i).memory
              << " " << std::endl;
  }
}

template <typename T>
void ConvFilterBackward(int algo, cudnnHandle_t handle,
                        cudnnTensorDescriptor_t x_desc,
                        cudnnTensorDescriptor_t y_desc,
                        cudnnFilterDescriptor_t w_desc,
                        cudnnConvolutionDescriptor_t conv_desc, T *x_ptr,
                        T *w_ptr, T *y_ptr, void *workspace,
                        size_t workspace_size) {
  float zero = 0.0;
  float one = 1.0;
  cudnnConvolutionBackwardFilter(
      handle, &one, x_desc, x_ptr, y_desc, y_ptr, conv_desc,
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(algo), workspace,
      workspace_size, &zero, w_desc, w_ptr);
}

template <typename T>
void ConvDataBackwardSearchAlgo(bool heuristic_search, cudnnHandle_t handle,
                                cudnnTensorDescriptor_t x_desc,
                                cudnnTensorDescriptor_t y_desc,
                                cudnnFilterDescriptor_t w_desc,
                                cudnnConvolutionDescriptor_t conv_desc,
                                T *x_ptr, T *w_ptr, T *y_ptr, void *workspace,
                                size_t workspace_size) {
  int count;
  cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &count);
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> res;
  res.resize(count);
  int ret;
  if (heuristic_search) {
    cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, w_desc, y_desc, conv_desc, x_desc, count, &ret, res.data());
  } else {
    cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle, w_desc, w_ptr, y_desc, y_ptr, conv_desc, x_desc, x_ptr, count,
        &ret, res.data(), workspace, workspace_size);
  }
  std::cout << "search backward data algorithm:" << std::endl;
  for (int i = 0; i < ret; ++i) {
    std::cout << "algo: " << res.at(i).algo
              << " mathtype: " << res.at(i).mathType
              << " time: " << res.at(i).time << " memory: " << res.at(i).memory
              << " " << std::endl;
  }
}

template <typename T>
void ConvDataBackward(int algo, cudnnHandle_t handle,
                      cudnnTensorDescriptor_t x_desc,
                      cudnnTensorDescriptor_t y_desc,
                      cudnnFilterDescriptor_t w_desc,
                      cudnnConvolutionDescriptor_t conv_desc, T *x_ptr,
                      T *w_ptr, T *y_ptr, void *workspace,
                      size_t workspace_size) {
  float zero = 0.0;
  float one = 1.0;
  cudnnConvolutionBackwardData(handle, &one, w_desc, w_ptr, y_desc, y_ptr,
                               conv_desc,
                               static_cast<cudnnConvolutionBwdDataAlgo_t>(algo),
                               workspace, workspace_size, &zero, x_desc, x_ptr);
}

template <typename T> void TestConv() {
  const int n = 32;
  const int x_c = 96;
  const int group = 96;
  const int y_c = 96;
  const int x_h = 56;
  const int x_w = 56;
  const int y_h = 56;
  const int y_w = 56;
  const int filter_in = x_c / group;
  const int filter_out = y_c;
  const int filter_size = 7;
  const int stride_h = 1;
  const int stride_w = 1;
  const int padding_h = 3;
  const int padding_w = 3;
  const int x_size = n * x_c * x_h * x_w * sizeof(T);
  const int w_size =
      filter_in * filter_out * filter_size * filter_size * sizeof(T);
  const int y_size = n * y_c * y_h * y_w * sizeof(T);
  const size_t workspace_size = 1024LL * 1024 * 1024 * 4;
  auto format = CUDNN_TENSOR_NCHW;
  cudnnDataType_t cuda_dtype = GetCudnnDataType<T>();
  cudnnDataType_t conv_desc_dtype = GetConvDescDataType<T>();
  cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;

  T *x;
  T *w;
  T *y;
  T *workspace;

  CudaCheck(cudaMalloc(&x, x_size));
  CudaCheck(cudaMalloc(&w, w_size));
  CudaCheck(cudaMalloc(&y, y_size));
  CudaCheck(cudaMalloc(&workspace, workspace_size));

  cudnnTensorDescriptor_t x_desc;
  CudaCheck(cudnnCreateTensorDescriptor(&x_desc));
  CudaCheck(
      cudnnSetTensor4dDescriptor(x_desc, format, cuda_dtype, n, x_c, x_h, x_w));

  cudnnTensorDescriptor_t y_desc;
  CudaCheck(cudnnCreateTensorDescriptor(&y_desc));
  CudaCheck(
      cudnnSetTensor4dDescriptor(y_desc, format, cuda_dtype, n, y_c, y_h, y_w));

  cudnnFilterDescriptor_t w_desc;
  CudaCheck(cudnnCreateFilterDescriptor(&w_desc));
  CudaCheck(cudnnSetFilter4dDescriptor(w_desc, cuda_dtype, format, filter_out,
                                       filter_in, filter_size, filter_size));

  cudnnConvolutionDescriptor_t conv_desc;
  CudaCheck(cudnnCreateConvolutionDescriptor(&conv_desc));
  CudaCheck(cudnnSetConvolution2dDescriptor(
      conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
      CUDNN_CROSS_CORRELATION, conv_desc_dtype));
  CudaCheck(cudnnSetConvolutionMathType(conv_desc, math_type));
  CudaCheck(cudnnSetConvolutionGroupCount(conv_desc, group));
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  cudnnHandle_t handle;
  CudaCheck(cudnnCreate(&handle));
  CudaCheck(cudnnSetStream(handle, stream));

  ConvForwardSearchAlgo(false, handle, x_desc, y_desc, w_desc, conv_desc, x, w,
                        y, workspace, workspace_size);
  ConvForward(0, handle, x_desc, y_desc, w_desc, conv_desc, x, w, y, workspace,
              workspace_size);

  ConvFilterBackwardSearchAlgo(false, handle, x_desc, y_desc, w_desc, conv_desc,
                               x, w, y, workspace, workspace_size);
  ConvFilterBackward(0, handle, x_desc, y_desc, w_desc, conv_desc, x, w, y,
                     workspace, workspace_size);

  ConvDataBackwardSearchAlgo(false, handle, x_desc, y_desc, w_desc, conv_desc,
                             x, w, y, workspace, workspace_size);
  ConvDataBackward(0, handle, x_desc, y_desc, w_desc, conv_desc, x, w, y,
                   workspace, workspace_size);
}

int main() { TestConv<nv_bfloat16>(); }
