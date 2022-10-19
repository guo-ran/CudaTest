#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <nccl.h>
#include <vector>
#include<thread>

void Check(ncclResult_t err) {
  if (err != ncclSuccess) {
    std::cerr << ncclGetErrorString(err) << std::endl;
    exit(1);
  }
}

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

constexpr int parallel_num = 4;
std::vector<int64_t> num_unique_matrix = {4384, 4474, 4505, 4443, 4402, 4570, 4377, 4413, 4456, 4532, 4474, 4479, 4487, 4455, 4425, 4460, 4499, 4556, 4595, 4568, 4520, 4533, 4569, 4535, 4475, 4482, 4507, 4401, 4395, 4436, 4419, 4435, 4509, 4420, 4596, 4444, 4548, 4528, 4502, 4475, 4362, 4384, 4493, 4375, 4483, 4344, 4447, 4451, 4524, 4478, 4445, 4383, 4479, 4468, 4424, 4373, 4441, 4377, 4541, 4481, 4520, 4478, 4451, 4442};
  

__global__ void LargeKernel(int64_t n, half* in_ptr, half* out_ptr) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    out_ptr[index] = (static_cast<float>(in_ptr[index]) + 1.f) / 2;
}

void KernelCompute(int gpu_id, ncclComm_t comm, cudaStream_t cuda_stream, half* send_data, half* recv_data, float* time, cudaEvent_t start_event, cudaEvent_t end_event) {
  Check(cudaSetDevice(gpu_id));
  std::vector<int64_t> send_elem_cnt;
  std::vector<int64_t> recv_elem_cnt;
  std::vector<int64_t> send_offsets;
  std::vector<int64_t> recv_offsets;
  int64_t send_cnt;
  int64_t recv_cnt;
  int64_t send_offset = 0;
  int64_t recv_offset = 0;
  for(int i=0;i<parallel_num;++i) {
    send_cnt = num_unique_matrix[gpu_id*parallel_num+i]*128;
    recv_cnt = num_unique_matrix[i*parallel_num+gpu_id]*128;
    send_elem_cnt.push_back(send_cnt);
    recv_elem_cnt.push_back(recv_cnt);
    send_offsets.push_back(send_offset);
    recv_offsets.push_back(recv_offset);
    send_offset += send_cnt;
    recv_offset += recv_cnt;
  }
  int64_t elem_cnt = 6912*26*128;
  for(int i=0;i<5000;++i) {
  LargeKernel<<<elem_cnt/1024, 1024, 0, cuda_stream>>>(elem_cnt, send_data, recv_data);
  }
  Check(cudaEventRecord(start_event, cuda_stream));
  Check(ncclGroupStart());
  for (int64_t i = 0; i < parallel_num; ++i) {
    Check(ncclSend(send_data + send_offsets.at(i), send_elem_cnt.at(i), ncclDataType_t::ncclFloat16, i,
                           comm, cuda_stream));
    Check(ncclRecv(recv_data + recv_offsets.at(i), recv_elem_cnt.at(i), ncclDataType_t::ncclFloat16, i,
                           comm, cuda_stream));
  }
  //for (int64_t k = 0; k < parallel_num; ++k) {
  //  //int i = (gpu_id + k + 1) % parallel_num;
  //  int i = (gpu_id + k) % parallel_num;
  //  Check(ncclSend(send_data + send_offsets.at(i), send_elem_cnt.at(i), ncclDataType_t::ncclFloat16, i,
  //                         comm, cuda_stream));
  //}
  //for (int64_t k = 0; k < parallel_num; ++k) {
  //  //int i = (gpu_id + parallel_num - 1 - k) % parallel_num;
  //  int i = (gpu_id + parallel_num - k) % parallel_num;
  //  Check(ncclRecv(recv_data + recv_offsets.at(i), recv_elem_cnt.at(i), ncclDataType_t::ncclFloat16, i,
  //                         comm, cuda_stream));
  //}
  Check(ncclGroupEnd());
  Check(cudaEventRecord(end_event, cuda_stream));
  Check(cudaEventSynchronize(end_event));
  Check(cudaEventElapsedTime(time, start_event, end_event));
}

int main() {
  int n_gpu = parallel_num;
  std::vector<cudaStream_t> streams(n_gpu);
  std::vector<ncclComm_t> comms(n_gpu);
  std::vector<cudaEvent_t> start_event(n_gpu);
  std::vector<cudaEvent_t> end_event(n_gpu);
  std::vector<half*> send_datas(n_gpu);
  std::vector<half*> recv_datas(n_gpu);
  std::vector<float> times(n_gpu);
  int64_t elem_cnt = 6912*26*128;
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(cudaStreamCreate(&streams.at(i)));
    Check(cudaMalloc(&send_datas.at(i), elem_cnt * sizeof(half)));
    Check(cudaMalloc(&recv_datas.at(i), elem_cnt * sizeof(half)));
    Check(cudaEventCreate(&start_event.at(i)));
    Check(cudaEventCreate(&end_event.at(i)));
  }
  ncclUniqueId unique_id;
  Check(ncclGetUniqueId(&unique_id));
  Check(ncclGroupStart());
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(ncclCommInitRank(&comms.at(i), n_gpu, unique_id, i));
  }
  Check(ncclGroupEnd());
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(cudaDeviceSynchronize());
  }
  std::thread threads[parallel_num];
  for (int i = 0; i < n_gpu; ++i) {
    threads[i] = std::thread(KernelCompute, i, comms.at(i), streams.at(i), send_datas.at(i), recv_datas.at(i), &times.at(i), start_event.at(i), end_event.at(i)); 
  }
  for (int i = 0; i < n_gpu; ++i) {
    threads[i].join();
    std::cout<<"times "<<times.at(i)<<std::endl;
  }
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(cudaDeviceSynchronize());
  }
  return 0;
}
