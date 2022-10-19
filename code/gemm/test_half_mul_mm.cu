

#include<cuda_fp16.h>
#include<fstream>
#include<iostream>
#include <cublas_v2.h>


template <typename T>
__global__ void NotPackDivKernel1(int64_t elem_cnt, T value, const T *in_a_ptr, const T *in_b_ptr,
                                 T *out_ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    out_ptr[i] = in_a_ptr[i] + in_b_ptr[i];
  }
}

template <typename T>
__global__ void NotPackDivKernel2(int64_t elem_cnt, T value, const T *in_a_ptr, const T *in_b_ptr,
                                 T *out_ptr) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < elem_cnt;
       i += gridDim.x * blockDim.x) {
    out_ptr[i] = in_a_ptr[i] + in_b_ptr[i];
  }
}

int main() {
    float mul=1/4.0;
    int m=6912;//mul*1024;
    int k=512;//1024;
    int n=256;//mul*1024;
    std::cout<<"m: "<<m<<" n: "<<n<<" k "<<k;
    int a_elem_cnt = m*k;
    int b_elem_cnt = k*n;
    int c_elem_cnt = m*n;
    half* in_a;
    half* in_b;
    half* out_c;
    half* in_a_2;
    half* in_b_2;
    half* out_c_2;

    cudaMalloc(&in_a, a_elem_cnt*sizeof(half));
    cudaMalloc(&in_b, b_elem_cnt*sizeof(half));
    cudaMalloc(&out_c, c_elem_cnt*sizeof(half));
    cudaMalloc(&in_a_2, a_elem_cnt*sizeof(half));
    cudaMalloc(&in_b_2, b_elem_cnt*sizeof(half));
    cudaMalloc(&out_c_2, c_elem_cnt*sizeof(half));

    half* in_a_host;
    cudaMallocHost(&in_a_host, a_elem_cnt*sizeof(half));
    half* in_b_host;
    cudaMallocHost(&in_b_host, b_elem_cnt*sizeof(half));
    half* out_c_host;
    cudaMallocHost(&out_c_host, c_elem_cnt*sizeof(half));

    void* workspace;
    size_t workspace_size = 4 * 1024 * 1024;
    cudaMalloc(&workspace, workspace_size);

    void* workspace2;
    cudaMalloc(&workspace2, workspace_size);

    std::ifstream x_is;
    x_is.open("in_a.bin");
    x_is.read(reinterpret_cast<char *>(in_a_host), a_elem_cnt*sizeof(half));
    x_is.close();
    x_is.open("in_b.bin");
    x_is.read(reinterpret_cast<char *>(in_b_host), b_elem_cnt*sizeof(half));
    x_is.close();
    cudaMemcpy(in_a, in_a_host, a_elem_cnt*sizeof(half), cudaMemcpyDefault);
    cudaMemcpy(in_b, in_b_host, b_elem_cnt*sizeof(half), cudaMemcpyDefault);
    
    cudaDeviceSynchronize();

    int least,greatest;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    std::cout<<"least "<<least<<" greatest "<<greatest<<std::endl;
    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    cudaStream_t cuda_stream;
    cudaStreamCreateWithPriority(&cuda_stream, cudaStreamDefault, greatest);
    //cudaStreamCreate(&cuda_stream);
    cublasSetStream(handle, cuda_stream);
    cublasSetWorkspace(handle, workspace, workspace_size);

    const float alpha_val = 1.0;
    const float beta_val = 0.0;
    cublasGemmEx(handle, cublasOperation_t::CUBLAS_OP_N,
                                     cublasOperation_t::CUBLAS_OP_N, n, m, k, &alpha_val, in_b, CUDA_R_16F, n, in_a,
                                     CUDA_R_16F, k, &beta_val, out_c, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    std::cout<<err;
    cudaMemcpy(out_c_host, out_c, c_elem_cnt*sizeof(half), cudaMemcpyDefault);
    cudaDeviceSynchronize();
    std::ofstream dx_os;
    dx_os.open("out_c.bin");
    dx_os.write(reinterpret_cast<char *>(out_c_host), c_elem_cnt*sizeof(half));
    dx_os.close();
    for(int i=0;i<10;++i){
      printf("out %f\n", static_cast<float>(out_c_host[i]));
    }
    return 0;
}
