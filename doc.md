
test_l2_cache 
```
/usr/local/cuda/bin/nvcc test_l2_cache.cu
/usr/local/cuda/nsight-systems-2020.4.3/bin/nsys profile --stat=true ./a.out
```
背景：
