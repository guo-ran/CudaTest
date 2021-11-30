第一步：
执行`python3 gen_random_data.py` 生成随机数据


Cudnn Softmax 
编译命令：
```
/usr/local/cuda-11.5/bin/nvcc cudnn_softmax_grad.cpp  -arch=sm_80 -O3 -I /usr/local/cudnn-11.5-linux-x64-v8.3.0.98/cuda/include/ -L/usr/local/cudnn-11.5-linux-x64-v8.3.0.98/cuda/lib64/ -lcudnn
```
ncu执行命令：
```
export LD_LIBRARY_PATH=/usr/local/cudnn-11.5-linux-x64-v8.3.0.98/cuda/lib64/:$LD_LIBRARY_PATH
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f  ./a.out 49152 256

```

OneFlow Softmax

编译命令：
```
/usr/local/cuda-11.5/bin/nvcc softmax_kernel_grad.cu -O3 -std=c++14 -arch=sm_80
```
ncu执行命令:
```
/usr/local/cuda-11.5/bin/ncu --section regex:'^(?!Nvlink)' -f  ./a.out 49152 256
```
