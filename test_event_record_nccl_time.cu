#include "cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <nccl.h>
#include <vector>

void Check(ncclResult_t err) {
  if (err != ncclSuccess) {
    std::cerr << ncclGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  int elem_cnt = 8 * 1024 * 1024;
  int n_gpu = 2;
  std::vector<cudaStream_t> streams(n_gpu);
  std::vector<half *> half_buffers(n_gpu);
  std::vector<float *> float_buffers(n_gpu);
  std::vector<ncclComm_t> comms(n_gpu);
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(cudaStreamCreate(&streams.at(i)));
    Check(cudaMalloc(&half_buffers.at(i), elem_cnt * sizeof(half)));
    Check(cudaMalloc(&float_buffers.at(i), elem_cnt * sizeof(float)));
    Check(cudaMemset(half_buffers.at(i), 0, elem_cnt * sizeof(half)));
    Check(cudaMemset(float_buffers.at(i), 0, elem_cnt * sizeof(float)));
  }

  Check(cudaDeviceSynchronize());
  ncclUniqueId unique_id;
  Check(ncclGetUniqueId(&unique_id));
  Check(ncclGroupStart());
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(ncclCommInitRank(&comms.at(i), n_gpu, unique_id, i));
  }
  Check(ncclGroupEnd());
  Check(cudaDeviceSynchronize());

  std::vector<cudaEvent_t> start_event(n_gpu);
  std::vector<cudaEvent_t> end_event(n_gpu);
  std::vector<float> time(n_gpu);
  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaSetDevice(i));
    Check(cudaEventCreate(&start_event.at(i)));
    Check(cudaEventCreate(&end_event.at(i)));
    Check(cudaEventRecord(start_event.at(i), streams.at(i)));
  }

  Check(ncclGroupStart());
  for (int i = 0; i < n_gpu; i++) {
    Check(cudaSetDevice(i));
    Check(ncclAllReduce(float_buffers.at(i), float_buffers.at(i), elem_cnt,
                        ncclDataType_t::ncclFloat, ncclRedOp_t::ncclSum,
                        comms.at(i), streams.at(i)));
    Check(ncclAllReduce(half_buffers.at(i), half_buffers.at(i), elem_cnt,
                        ncclDataType_t::ncclFloat16, ncclRedOp_t::ncclSum,
                        comms.at(i), streams.at(i)));
  }
  Check(ncclGroupEnd());

  for (int i = 0; i < n_gpu; ++i) {
    Check(cudaEventRecord(end_event.at(i), streams.at(i)));
    Check(cudaEventSynchronize(end_event.at(i)));
    Check(cudaEventElapsedTime(&time.at(i), start_event.at(i), end_event.at(i)));
    std::cout<<"time "<<i <<" : "<<time.at(i)<<std::endl;
  }
  Check(cudaDeviceSynchronize());

  return 0;
}
