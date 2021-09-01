/*

/usr/local/cuda-11.4/bin/nvcc test_nccl.cu -I /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/include/ -L /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ -lnccl 
export LD_LIBRARY_PATH=/usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ 

/usr/local/cuda-11.4/bin/nvcc test_nccl.cu -I /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/include/ -L /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ -lnccl_static

 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <nccl.h>
#include <iostream>
#include <vector>

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

void Check(ncclResult_t err) {
  if (err != ncclSuccess) {
    std::cerr << ncclGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  int elem_cnt = 4 * 1024 * 1024;
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
  Check(cudaDeviceSynchronize());

  return 0;
}
