#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

template <typename T>
__global__ void CopyKernel0(T *in, T *out, int rows, int cols) {
  int tid = threadIdx.x;
  const int f2_cols = cols / 2;
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    for (int col = tid; col < f2_cols; col += blockDim.x) {
      float2 f2;
      const int64_t offset = row * cols + col * 2;
      f2 = *reinterpret_cast<const float2 *>(in + offset);
      f2.x = f2.x - 1.0f;
      f2.y = f2.y - 1.0f;
      *reinterpret_cast<float2 *>(out + offset) = f2;
    }
  }
}

template <typename T>
__global__ void CopyKernel1(T *in, T *out, int rows, int cols) {
  int tid = threadIdx.x;
  const int f2_cols = cols / 2;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    for (int col = tid; col < f2_cols; col += blockDim.x) {
      float2 f2;
      const int64_t offset = row * cols + col * 2;
      f2 = *reinterpret_cast<const float2 *>(in + offset);
      f2.x = f2.x - 1.0f;
      f2.y = f2.y - 1.0f;
      *reinterpret_cast<float2 *>(out + offset) = f2;
    }
  }
}

int main() {
  float *in;
  float *out;
  int num_rows = 8192;
  int num_cols = 4096;
  int data_size = num_rows * num_cols * sizeof(float);
  Check(cudaMalloc(&in, data_size));
  Check(cudaMalloc(&out, data_size));
  CopyKernel0<float><<<2048, 512>>>(in, out, num_rows, num_cols);
  CopyKernel1<float><<<2048, 512>>>(in, out, num_rows, num_cols);
  Check(cudaDeviceSynchronize());
  Check(cudaFree(in));
  Check(cudaFree(out));
  return 0;
}
