#include "cuda_util.h"
#include <cuda_runtime.h>

__global__ void Relu1(const int64_t n, float *x, float *y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float x_i = x[i];
    float y_i = x_i > 0 ? x_i : 0;
    y[i] = y_i;
  }
}

__global__ void Relu2(const int64_t n, float *x, float *y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float x_i = x[i];
    float y_i = x_i > 0 ? x_i : 0;
    y[i] = y_i;
  }
}

__global__ void Relu3(const int64_t n, float *x, float *y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float x_i = x[i];
    float y_i = x_i > 0 ? x_i : 0;
    y[i] = y_i;
  }
}

int main(int argc, char *argv[]) {
  int data_size = 4 * 1024 * 1024;
  float *x;
  float *y;
  float *z;
  float *host_in;
  float *host_out;
  Check(cudaMalloc(&x, data_size * sizeof(float)));
  Check(cudaMalloc(&y, data_size * sizeof(float)));
  Check(cudaMalloc(&z, data_size * sizeof(float)));
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  host_in = (float *)calloc(data_size, sizeof(float));
  host_out = (float *)calloc(data_size, sizeof(float));
  for (int i = 0; i < data_size; ++i) {
    host_in[i] = i;
  }

  Check(cudaMemcpy(x, host_in, data_size * sizeof(float), cudaMemcpyDefault));
  Check(cudaMemcpy(z, host_in, data_size * sizeof(float), cudaMemcpyDefault));
  Check(cudaMemcpy(z, host_in, data_size * sizeof(float), cudaMemcpyDefault));
  Check(cudaMemcpy(z, host_in, data_size * sizeof(float), cudaMemcpyDefault));
  cudaDeviceProp prop;               // CUDA device properties variable
  cudaGetDeviceProperties(&prop, 0); // Query GPU properties
  size_t size = min(int(prop.l2CacheSize * 1.0), prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                     size); // set-aside 3/4 of L2 cache for persisting accesses
                            // or the max allowed

  int num_bytes = 134213632; // 10*1024*1024;
  size_t window_size = min(prop.accessPolicyMaxWindowSize,
                           num_bytes); // Select minimum of user defined
                                       // num_bytes and max window size.
  std::cout << "window_size " << window_size
            << " prop.l2CacheSize: " << prop.l2CacheSize
            << " prop.persistingL2CacheMaxSize: "
            << prop.persistingL2CacheMaxSize
            << "  prop.accessPolicyMaxWindowSize "
            << prop.accessPolicyMaxWindowSize << std::endl;
  {
    cudaStreamAttrValue
        stream_attribute; // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr =
        reinterpret_cast<void *>(y); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes =
        window_size; // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio =
        1.0; // float(size) / float(num_bytes); // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp =
        cudaAccessPropertyPersisting; // Persistence Property
    stream_attribute.accessPolicyWindow.missProp =
        cudaAccessPropertyNormal; // Type of access property on cache miss
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attribute);
  }

  Relu1<<<4096, 512, 0, stream>>>(data_size, x, y);
  // stream_attribute.accessPolicyWindow.num_bytes = 0; // Setting the window
  // size to 0 disable it cudaStreamSetAttribute(stream,
  // cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite
  // the access policy attribute to a CUDA Stream
  // cudaCtxResetPersistingL2Cache();

  { // Stream level attributes data structure
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr =
        reinterpret_cast<void *>(z); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes =
        window_size; // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio =
        1.0; // float(size) / float(num_bytes) * 0.75; // Hint for cache hit
             // ratio
    stream_attribute.accessPolicyWindow.hitProp =
        cudaAccessPropertyPersisting; // Persistence Property
    stream_attribute.accessPolicyWindow.missProp =
        cudaAccessPropertyNormal; // Type of access property on cache miss
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attribute);
  }

  Relu2<<<4096, 512, 0, stream>>>(data_size, y, z);
  Relu3<<<4096, 512, 0, stream>>>(data_size, z, x);

  cudaStreamSynchronize(stream);

  Check(cudaMemcpy(host_out, y, data_size * sizeof(float),
                   cudaMemcpyDeviceToHost));
  return 0;
}
