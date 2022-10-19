import oneflow as flow
import numpy as np
import math


batch_size = 64
num_head = 16
max_seqlen_q = 1024
max_seqlen_k = 1024
head_size = 1024 // num_head
np_dtype = np.float16
np_q = np.fromfile("q.bin",np.float16).reshape(batch_size, max_seqlen_q, num_head, head_size).transpose(0, 2, 1, 3)
np_k = np.fromfile("k.bin",np.float16).reshape(batch_size, max_seqlen_k, num_head, head_size).transpose(0, 2, 1, 3)
np_v = np.fromfile("v.bin",np.float16).reshape(batch_size, max_seqlen_k, num_head, head_size).transpose(0, 2, 1, 3)

q = flow.tensor(np_q, device="cuda", requires_grad=True)
k = flow.tensor(np_k, device="cuda", requires_grad=True)
v = flow.tensor(np_v, device="cuda", requires_grad=True)

dropout_p=0.0
scores = flow.matmul(q, k, transpose_b=True, alpha=1/math.sqrt(head_size))
#attention_weights = flow._C.fused_scale_tril_softmax_mask_scale(
#                scores,
#                p=dropout_p,
#                diagonal=0,
#                tril_scale_value=1.0,
#                tril_fill_value=-10000.0,
#            )[0]
attention_weights = flow.softmax(scores, dim=-1)
np.save("softmax_out", attention_weights)
out = flow.matmul(attention_weights, v)
out = out.permute([0, 2, 1, 3])
loss = out.sum()
loss.backward()
out_host = np.fromfile("out.bin",np.float16).reshape(batch_size, max_seqlen_k, num_head, head_size)
print(np.allclose(out_host, out.numpy(), 1e-3, 1e-3))
print(out.numpy().flatten()[0:10])
dq_host = np.fromfile("dq.bin",np.float16).reshape(batch_size, max_seqlen_q, num_head, head_size).transpose(0, 2, 1, 3)
print(np.allclose(dq_host, q.grad.numpy(), 1e-3, 1e-3))
print("dq_host", dq_host.flatten()[0:10])
print("q_grad", q.grad.numpy().flatten()[0:10])
dk_host = np.fromfile("dk.bin",np.float16).reshape(batch_size, max_seqlen_q, num_head, head_size).transpose(0, 2, 1, 3)
print(np.allclose(dk_host, k.grad.numpy(), 1e-3, 1e-3))
print("dk_host", dk_host.flatten()[0:10])
print("k_grad", k.grad.numpy().flatten()[0:10])
dv_host = np.fromfile("dv.bin",np.float16).reshape(batch_size, max_seqlen_q, num_head, head_size).transpose(0, 2, 1, 3)
print(np.allclose(dv_host, v.grad.numpy(), 1e-3, 1e-3))
print("dv_host", dv_host.flatten()[0:10])
print("v_grad", v.grad.numpy().flatten()[0:10])
