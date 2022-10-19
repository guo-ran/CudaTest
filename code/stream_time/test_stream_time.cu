#include<iostream>


void CudaCheck(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void LargeKernel(int64_t n, float* in_ptr, float* out_ptr) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    out_ptr[index] = (in_ptr[index] + 1.f) / 2;
}
__global__ void Kernel(int64_t n, float* in_ptr, float* out_ptr, float* in_ptr1, float* out_ptr1) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    out_ptr[index] = (in_ptr[index] + 1.f) / 2;
}

int main() {
    int64_t elem_cnt = 64*1024*1024;
    int64_t small_elem_cnt = 10*1024*1024;
    float* in_ptr;
    float* out_ptr;
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    cudaStream_t stream2;
    CudaCheck(cudaStreamCreate(&stream2));
    CudaCheck(cudaMalloc(&in_ptr, elem_cnt*sizeof(float)));
    CudaCheck(cudaMalloc(&out_ptr, elem_cnt*sizeof(float)));
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaEvent_t stream1_event;
    cudaEvent_t stream2_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventCreate(&stream1_event);
    cudaEventCreate(&stream2_event);  
    bool use_two_stream = true;  
    bool use_cuda_graph = true;
    cudaGraphExec_t graph_exec;
    cudaGraph_t graph = nullptr;
    if(use_cuda_graph) {
        CudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<100;i++) {
          Kernel<<<small_elem_cnt/1024, 1024, 0, stream>>>(small_elem_cnt, in_ptr, out_ptr, in_ptr, out_ptr);
          if(use_two_stream) {
            CudaCheck(cudaEventRecord(stream1_event, stream));
            CudaCheck(cudaStreamWaitEvent(stream2, stream1_event));
            Kernel<<<small_elem_cnt/1024, 1024, 0, stream2>>>(small_elem_cnt, out_ptr, in_ptr, in_ptr, out_ptr);
            CudaCheck(cudaEventRecord(stream2_event, stream2));
            CudaCheck(cudaStreamWaitEvent(stream, stream2_event));
          } else {
            Kernel<<<small_elem_cnt/1024, 1024, 0, stream>>>(small_elem_cnt, out_ptr, in_ptr, in_ptr, out_ptr);
          }
        }
        CudaCheck(cudaStreamEndCapture(stream, &graph));
        CudaCheck(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
        CudaCheck(cudaGraphDestroy(graph));
    }
    LargeKernel<<<elem_cnt/1024, 1024, 0, stream>>>(elem_cnt, in_ptr, out_ptr);
    CudaCheck(cudaEventRecord(start_event, stream));
    if(use_cuda_graph) {
        CudaCheck(cudaGraphLaunch(graph_exec, stream));
    } else {
        for(int i=0;i<100;i++) {
          Kernel<<<small_elem_cnt/1024, 1024, 0, stream>>>(small_elem_cnt, in_ptr, out_ptr, in_ptr, out_ptr);
          if(use_two_stream) {
            CudaCheck(cudaEventRecord(stream1_event, stream));
            CudaCheck(cudaStreamWaitEvent(stream2, stream1_event));
            Kernel<<<small_elem_cnt/1024, 1024, 0, stream2>>>(small_elem_cnt, out_ptr, in_ptr, in_ptr, out_ptr);
            CudaCheck(cudaEventRecord(stream2_event, stream2));
            CudaCheck(cudaStreamWaitEvent(stream, stream2_event));
          } else {
            Kernel<<<small_elem_cnt/1024, 1024, 0, stream>>>(small_elem_cnt, out_ptr, in_ptr, in_ptr, out_ptr);
          }
        }
    }
    CudaCheck(cudaEventRecord(end_event, stream));
    CudaCheck(cudaEventSynchronize(end_event));
    float time;
    CudaCheck(cudaEventElapsedTime(&time, start_event, end_event));
    std::cout<<"time "<<time<<std::endl;
    return 0;
}
