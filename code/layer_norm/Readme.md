第一步：
执行`python3 gen_random_data.py` 生成随机数据

Apex LayerNorm，修改layer_norm_cuda_kernel.cu文件里main函数可以测试前后向，float/half类型，目前后向里beta传了nullptr，所以后向只有dx的kernel
编译命令：
```
/usr/local/cuda-11.5/bin/nvcc apex_layer_norm.cu -O3 -std=c++14 -arch=sm_80 --use_fast_math
```
ncu执行命令：
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f ./a.out 49152 32

```

OneFlow LayerNorm前向, 注意需要define OF_LAYER_NORM_USE_FAST_MATH，修改main函数里可以测试float/half类型。

编译命令：
```
/usr/local/cuda-11.5/bin/nvcc of_layer_norm.cu -O3 -std=c++14 -arch=sm_80
```
ncu执行命令:
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f  ./a.out 49152 256
```

OneFlow LayerNorm后向，修改main函数里可以测试float/half类型。

编译命令：
```
/usr/local/cuda-11.5/bin/nvcc of_layer_norm_grad.cu -O3 -std=c++14 -arch=sm_80
```
ncu执行命令:
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f  ./a.out 49152 256
```

Pytorch LayerNorm前向，只支持float类型。

```
import torch
import torch.nn as nn
import sys
norm_size=int(sys.argv[1])

x=torch.randn(49152, norm_size).to(torch.float).to("cuda")
layernorm=nn.LayerNorm((norm_size)).to("cuda")

print(x.shape, x.dtype)
out=layernorm(x)
print(out)
```

我编译cuda代码测试的PyTorch比ncu执行python RowwiseMomentsCUDAKernel测出的时间慢200us(col=32下 python 317.60 vs cuda kernel 500)，目前不清楚为什么。可能是缺了编译参数
目前编译参数
```
/usr/local/cuda-11.2/bin/nvcc pytorch_layer_norm.cu -O3 -std=c++14 -arch=sm_80 --use_fast_math
```
