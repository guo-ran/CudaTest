

#include<cuda_fp16.h>
#include<fstream>
#include<iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}



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
    int m=8192;//mul*1024;
    int k=8192;//1024;
    int n=4096;//mul*1024;
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
    size_t workspace_size = 16 * 1024 * 1024;
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
    cudaStream_t cuda_stream;
    //cudaStreamCreateWithPriority(&cuda_stream, cudaStreamDefault, greatest);
    cudaStreamCreate(&cuda_stream);
    cudaStream_t cuda_stream2;
    cudaStreamCreate(&cuda_stream2);
    //cudaStreamCreateWithPriority(&cuda_stream2, cudaStreamDefault, least);

    float time1,time2;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaEvent_t end_event2;
    Check(cudaEventCreate(&start_event));
    Check(cudaEventCreate(&end_event));
    Check(cudaEventCreate(&end_event2));
    Check(cudaEventRecord(start_event, cuda_stream));
    NotPackDivKernel1<half><<<216*2, 512, 0, cuda_stream>>>(
      m*k, static_cast<half>(2), in_a, in_b, out_c);
    NotPackDivKernel1<half><<<216*2, 512, 0, cuda_stream>>>(
      m*k, static_cast<half>(2), in_a, in_b, out_c);
    NotPackDivKernel1<half><<<216*2, 512, 0, cuda_stream>>>(
      m*k, static_cast<half>(2), in_a, in_b, out_c);
    NotPackDivKernel1<half><<<216*2, 512, 0, cuda_stream>>>(
      m*k, static_cast<half>(2), in_a, in_b, out_c);
    Check(cudaEventRecord(end_event, cuda_stream));
    //cudaStreamWaitEvent(cuda_stream2, start_event);
    //NotPackDivKernel2<half><<<216, 512, 0, cuda_stream2>>>(
    //  m*k, static_cast<half>(2), in_a_2, in_b_2, out_c_2);
    //NotPackDivKernel2<half><<<216, 512, 0, cuda_stream2>>>(
    //  m*k, static_cast<half>(2), in_a_2, in_b_2, out_c_2);
    //Check(cudaEventRecord(end_event2, cuda_stream2));
    Check(cudaEventSynchronize(end_event));
    //Check(cudaEventSynchronize(end_event2));
    Check(cudaEventElapsedTime(&time1, start_event, end_event));
    //Check(cudaEventElapsedTime(&time2, start_event, end_event2));
    std::cout<<"time1 "<<time1<<std::endl;
    std::cout<<"time2 "<<time2<<std::endl;

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
      //printf("out %f\n", static_cast<float>(out_c_host[i]));
    }
    return 0;
}
