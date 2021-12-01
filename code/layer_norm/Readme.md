第一步：
执行`python3 gen_random_data.py` 生成随机数据


Apex LayerNorm，修改layer_norm_cuda_kernel.cu文件测试前后向，目前后向里beta传了nullptr，所以后向只有dx的kernel
编译命令：
```
/usr/local/cuda-11.5/bin/nvcc layer_norm_cuda_kernel.cu -O3 -std=c++14 -arch=sm_80 --use_fast_math
```
ncu执行命令：
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f ./a.out 49152 32

```

OneFlow LayerNorm前向, 注意需要define OF_LAYER_NORM_USE_FAST_MATH

编译命令：
```
/usr/local/cuda-11.5/bin/nvcc layer_norm_kernel.cu -O3 -std=c++14 -arch=sm_80
```
ncu执行命令:
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f  ./a.out 49152 256
```

OneFlow LayerNorm后向

编译命令：
```
/usr/local/cuda-11.5/bin/nvcc layer_norm_kernel_grad.cu -O3 -std=c++14 -arch=sm_80
```
ncu执行命令:
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f  ./a.out 49152 256
```
