# cuda_test

/usr/local/cuda-11.4/bin/nvcc test_nccl.cu -I /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/include/ -L /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ -lnccl
export LD_LIBRARY_PATH=/usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/
/usr/local/cuda-11.4/bin/nvcc test_nccl.cu -I /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/include/ -L /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ -lnccl_static
test_nccl.cu is test nccl allreduce half float group call.
