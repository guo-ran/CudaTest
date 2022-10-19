rm report*
#export NCCL_BUFFSIZE=8388608 #16777216 #8388608
#export NCCL_MIN_NCHANNELS=128
#export NCCL_CHECKS_DISABLE=1
#export NCCL_SINGLE_RING_THRESHOLD=524288
#export NCCL_SET_STACK_SIZE=1
export NCCL_LL_THRESHOLD=32768
//usr/local/cuda-11.7/bin/nsys profile --stat=true ./a.out
