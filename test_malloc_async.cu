#include <iostream>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

template<typename T>
__global__ void Fill(int n, T* out) {
    for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
        out[i] = 1.5;
    }
}

template<typename T>
__global__ void Copy(int n, T* in, T* out) {
    for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
        out[i] = in[i] + 1.2;
    }
}

int main() {
    using T = float;
    cudaEvent_t event;
    Check(cudaEventCreate(&event));
    cudaStream_t stream_a;
    cudaStream_t stream_b;
    Check(cudaStreamCreate(&stream_a));
    Check(cudaStreamCreate(&stream_b));
    T* in_ptr=nullptr;
    T* out_ptr;
    T* host_out_ptr;
    int elem_cnt = 1024 * 1024 * 4;
    size_t size = elem_cnt * sizeof(T);
    cudaMallocHost(&host_out_ptr, size);
    //cudaMemPool_t mempool;
    //cudaDeviceGetDefaultMemPool(&mempool, 0);
    //uint64_t threshold = UINT64_MAX;
    //cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    for(int iter=0;iter<1;iter++) {
        Check(cudaMallocAsync(&in_ptr, size, stream_a));
        Fill<<<elem_cnt/1024, 1024, 0, stream_a>>>(elem_cnt, in_ptr);
        Check(cudaEventRecord(event, stream_a));
        Check(cudaStreamWaitEvent(stream_b, event, 0));
        Check(cudaMallocAsync(&out_ptr, size, stream_b));
        Copy<<<elem_cnt/1024, 1024, 0, stream_b>>>(elem_cnt, in_ptr, out_ptr);
        Check(cudaFreeAsync(in_ptr, stream_b));
        cudaMemcpyAsync(host_out_ptr, out_ptr, size, cudaMemcpyDefault, stream_b);
        Check(cudaFreeAsync(out_ptr, stream_b));
        cudaStreamSynchronize(stream_b);
    }
    for(int i=0;i<10;++i) {
        std::cout<<"out "<<i<<" "<<host_out_ptr[i]<<std::endl;
    }
    Check(cudaStreamDestroy(stream_a));
    Check(cudaStreamDestroy(stream_b));
    Check(cudaEventDestroy(event));
}
