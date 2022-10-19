import torch
import numpy as np
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
batch_size = 64
nheads = 16
seqlen = 1024
n = 1024
d = n // nheads
dropout_p = 0
causal = False
dtype = torch.float16
device = 'cuda'
q = torch.tensor(np.fromfile("q.bin",np.float16).reshape(batch_size*seqlen, nheads, d), device='cuda', dtype=dtype, requires_grad=True)
k = torch.tensor(np.fromfile("k.bin",np.float16).reshape(batch_size*seqlen, nheads, d), device='cuda', dtype=dtype, requires_grad=True)
v = torch.tensor(np.fromfile("v.bin",np.float16).reshape(batch_size*seqlen, nheads, d), device='cuda', dtype=dtype, requires_grad=True)
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device='cuda')
out, softmax_lse, S_dmask = flash_attn_unpadded_func(q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen,
                             dropout_p, softmax_scale=None, causal=False, return_attn_probs=True)
grad = torch.ones_like(out)
out.backward(grad, retain_graph=True)
print("out", out[0][0])
print("q grad", q.grad[0][0])
print("k grad", k.grad[0][0])
print("v grad", v.grad[0][0])
out_host = np.fromfile("out.bin",np.float16).reshape(batch_size*seqlen, nheads, d)
print("allclose out", np.allclose(out_host, out.detach().cpu().numpy(), 1e-3, 1e-3))
softmax_lse_host = np.fromfile("softmax_lse.bin",np.float32).reshape(batch_size, nheads, seqlen)
print("allclose softmax_lse", np.allclose(softmax_lse_host, softmax_lse.detach().cpu().numpy(), 1e-3, 1e-3))
s_dmask_host = np.fromfile("softmax.bin",np.float16).reshape(batch_size, nheads, seqlen, seqlen)
print("allclose S_dmask", np.allclose(s_dmask_host[0][0], S_dmask.detach().cpu().numpy()[0][0], 1e-3, 1e-3))
print("S_dmask", s_dmask_host[0][0][0])
print("S_dmask", S_dmask.detach().cpu().numpy()[0][0][0])
