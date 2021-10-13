#include<cuda.h>
#include<algorithm>

const int32_t kCudaThreadsNumPerBlock = 512;

// elem_cnt % kCudaThreadsNumPerBlock == 0, elem_cnt < max_num_block
// A100 sm80, 3090 sm86, 2080Ti sm75, V100 sm70  
// /usr/local/cuda/bin/nvcc test_copy_benchmark.cu -arch=sm_80 -O3 -std=c++11
// /usr/local/cuda/bin/ncu --section ".*" --target-processes all -f ./a.out > copy_benchmark
// /usr/local/cuda-11.4/bin/ncu --section regex:'^(?!Nvlink)' -f ./a.out > copy_benchmark
// cat copy_benchmark | grep "Memory Throughput\|Copy"

template<int mb, int N>
__global__ void CopyKernel(const void* in, void* out) {
    using T = typename std::aligned_storage<N, N>::type;
    const T* src = reinterpret_cast<const T*>(in);
    T* dst = reinterpret_cast<T*>(out);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i];
}

template<int mb, int N>
void LaunchCopyKernel(const void* in, void* out) {
    CopyKernel<mb, N><<<mb*1024*1024/N/kCudaThreadsNumPerBlock, kCudaThreadsNumPerBlock, 0>>>(in, out); 
}

int main() {
    int max_size = 1024*1024*1024;
    void* in;
    void* out;
    cudaMalloc(&in, max_size);
    cudaMalloc(&out, max_size);

    LaunchCopyKernel<1, 1>(in, out); 
    LaunchCopyKernel<2, 1>(in, out); 
    LaunchCopyKernel<4, 1>(in, out); 
    LaunchCopyKernel<8, 1>(in, out); 
    LaunchCopyKernel<16, 1>(in, out); 
    LaunchCopyKernel<32, 1>(in, out); 
    LaunchCopyKernel<64, 1>(in, out); 
    LaunchCopyKernel<128, 1>(in, out); 
    LaunchCopyKernel<256, 1>(in, out); 
    LaunchCopyKernel<512, 1>(in, out); 
    LaunchCopyKernel<1024, 1>(in, out); 

    LaunchCopyKernel<1, 2>(in, out); 
    LaunchCopyKernel<2, 2>(in, out); 
    LaunchCopyKernel<4, 2>(in, out); 
    LaunchCopyKernel<8, 2>(in, out); 
    LaunchCopyKernel<16, 2>(in, out); 
    LaunchCopyKernel<32, 2>(in, out); 
    LaunchCopyKernel<64, 2>(in, out); 
    LaunchCopyKernel<128, 2>(in, out); 
    LaunchCopyKernel<256, 2>(in, out); 
    LaunchCopyKernel<512, 2>(in, out); 
    LaunchCopyKernel<1024, 2>(in, out); 

    LaunchCopyKernel<1, 4>(in, out); 
    LaunchCopyKernel<2, 4>(in, out); 
    LaunchCopyKernel<4, 4>(in, out); 
    LaunchCopyKernel<8, 4>(in, out); 
    LaunchCopyKernel<16, 4>(in, out); 
    LaunchCopyKernel<32, 4>(in, out); 
    LaunchCopyKernel<64, 4>(in, out); 
    LaunchCopyKernel<128, 4>(in, out); 
    LaunchCopyKernel<256, 4>(in, out); 
    LaunchCopyKernel<512, 4>(in, out); 
    LaunchCopyKernel<1024, 4>(in, out); 

    LaunchCopyKernel<1, 8>(in, out); 
    LaunchCopyKernel<2, 8>(in, out); 
    LaunchCopyKernel<4, 8>(in, out); 
    LaunchCopyKernel<8, 8>(in, out); 
    LaunchCopyKernel<16, 8>(in, out); 
    LaunchCopyKernel<32, 8>(in, out); 
    LaunchCopyKernel<64, 8>(in, out); 
    LaunchCopyKernel<128, 8>(in, out); 
    LaunchCopyKernel<256, 8>(in, out); 
    LaunchCopyKernel<512, 8>(in, out); 
    LaunchCopyKernel<1024, 8>(in, out); 

    LaunchCopyKernel<1, 16>(in, out); 
    LaunchCopyKernel<2, 16>(in, out); 
    LaunchCopyKernel<4, 16>(in, out); 
    LaunchCopyKernel<8, 16>(in, out); 
    LaunchCopyKernel<16, 16>(in, out); 
    LaunchCopyKernel<32, 16>(in, out); 
    LaunchCopyKernel<64, 16>(in, out); 
    LaunchCopyKernel<128, 16>(in, out); 
    LaunchCopyKernel<256, 16>(in, out); 
    LaunchCopyKernel<512, 16>(in, out); 
    LaunchCopyKernel<1024, 16>(in, out); 

    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);
    return 0;
}

