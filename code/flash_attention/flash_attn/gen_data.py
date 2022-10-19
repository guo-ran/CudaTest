import oneflow as flow
import numpy as np


batch_size = 64
num_head = 16
max_seqlen_q = 1024
max_seqlen_k = 1024
head_size = 1024 // num_head
np_dtype = np.float16
q = np.random.rand(batch_size, num_head, max_seqlen_q, head_size).astype(np_dtype)
k = np.random.rand(batch_size, num_head, max_seqlen_k, head_size).astype(np_dtype)
v = np.random.rand(batch_size, num_head, max_seqlen_k, head_size).astype(np_dtype)

q.tofile("q.bin")
k.tofile("k.bin")
v.tofile("v.bin")
