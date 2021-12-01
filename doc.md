



#### test_l2_cache 

编译命令：

```
/usr/local/cuda/bin/nvcc test_l2_cache.cu
/usr/local/cuda/nsight-systems-2020.4.3/bin/nsys profile --stat=true ./a.out
```
背景：

Cuda 11开始，8.0架构开始可以控制驻留L2 cache的数据， [L2 cache](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#L2-cache)

test_l2_cache用于测试设置驻留L2 cache数据的效果。

在Relu1调用前设置了10M cache 驻留保存y指针的数据。Relu2用y作为in，所以Relu2会有加速。在Relu2调用前设置了10M cache驻留保存z指针的数据，所以Relu3会有加速。

```
stream_attribute.accessPolicyWindow.hitRatio 代表给定一段数据有多大概率命中
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; 代表命中的行为，是驻留在L2cache中
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal； 代表未命中的行为
```

这段代码在A100 PCIE 40GB上的结果是：

```
 Time(%)  Total Time (ns)   Average   Minimum  Maximum             Name
 -------  ---------------   --------  -------  -------  ---------------------------
    35.6           34,144    34,144.0   34,144   34,144  Relu1(long, float*, float*)
    33.6           32,193    32,193.0   32,193   32,193  Relu3(long, float*, float*)
    30.7           29,440    29,440.0   29,440   29,440  Relu2(long, float*, float*)
```

Relu2和Relu3相对Relu1有加速，因为数据驻留在L2 cache中。

这些设置在3090上没有效果，在A100上有效果。



#### test_nccl

编译命令：

```
/usr/local/cuda-11.4/bin/nvcc test_nccl.cu -I /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/include/ -L /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ -lnccl
export LD_LIBRARY_PATH=/usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/

/usr/local/cuda-11.4/bin/nvcc test_nccl.cu -I /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/include/ -L /usr/local/nccl_2.10.3-1+cuda11.4_x86_64/lib/ -lnccl_static

test_nccl.cu is test nccl allreduce half float group call.
```

背景：

nccl2.10.3支持了bfloat16数据类型，在我更新了nccl 版本后，发现resnet50网络中allreduce部分有bfloat16会不收敛，出现nan。此外发现这个版本float16也不收敛了，实验排查确认是nccl版本的问题。此外还有现象是采用cpu decoder，会在collective boxing executor后报700内存访问的错误。不运行不同数据类型fuse进一个group就不会发生这些情况。

本测试文件是构造的2卡的混合数据类型的allreduce，一个float类型，一个float16类型，nccl版本2.10.3在A100上动态链接会报`an illegal memory access was encountered`错误，nccl版本2.9.8没问题，向nccl官方提了issue，已fix，等待下一个版本发布。

#### test_copy_benchmark
编译命令：
```
A100 sm_80, 3090 sm86, 2080Ti sm75, V100 sm70
/usr/local/cuda-11.4/bin/nvcc test_copy_benchmark.cu -arch=sm_80 -O3 -std=c++11
```
运行ncu命令：
```
cuda 11.2:
sudo /usr/local/cuda-11.2/bin/ncu --section ".*" -f ./a.out
```

cuda 11.4命令发生了变化，直接跑上述命令会报错：

```
==ERROR== Option '--section .*' did not match any section.
```

查文档，将`--section .*`改成`--section "regex:.*"` 后，会报错：

```
==ERROR== Failed to access the following 8 metrics: nvlrx__bytes.sum, nvlrx__bytes.sum.pct_of_peak_sustained_elapsed, nvlrx__bytes_data_protocol.sum, nvlrx__bytes_data_user.sum, nvltx__bytes.sum, nvltx__bytes.sum.pct_of_peak_sustained_elapsed, nvltx__bytes_data_protocol.sum, nvltx__bytes_data_user.sum
```

原因是这台机器没有nvlink，需要改成如下命令：

```
cuda 11.4:
/usr/local/cuda-11.4/bin/ncu --section regex:'^(?!Nvlink)' -f ./a.out
```

背景：  

使用CopyKernel测试不同数据大小、不同pack_size下，能达到的显存带宽上限，作为Cuda Kernel优化的依据。详见[Copy-Benchmark](doc/copy_benchmark.md)


#### test_softmax

cudnn 8.3 的softmax速度快了不少，基于cudnn8.3做了和oneflow性能测试比较，详见[Softmax](doc/softmax.md)

#### test_layer_norm

基于apex做了和oneflow性能测试比较，详见[LayerNorm](doc/layer_norm.md)





