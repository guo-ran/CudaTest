import oneflow as flow
import numpy as np

batch_size = 55296 // 8
embedding_size = 128
np_dtype = np.float16
in_0 = np.random.rand(batch_size, 1, embedding_size).astype(np_dtype)
in_1 = np.random.rand(batch_size, 26, embedding_size).astype(np_dtype)
output_concat = np.random.rand(batch_size, embedding_size).astype(np_dtype)
in_0_tensor = flow.tensor(in_0, device="cuda", requires_grad=True)
output_concat_tensor = flow.tensor(output_concat, device="cuda", requires_grad=True)
in_1_tensor = flow.tensor(in_1, device="cuda", requires_grad=True)
out_tensor = flow._C.fused_dot_feature_interaction(
    [in_0_tensor, in_1_tensor],
    output_concat=output_concat_tensor,
    self_interaction=False,
    output_padding=1,
    pooling="none",
)
fused_loss = out_tensor.sum()
fused_loss.backward()
feature_0_grad = in_0_tensor.grad.numpy()
output_concat_grad = output_concat_tensor.grad.numpy()
feature_1_grad = in_1_tensor.grad.numpy()
dy = np.ones(out_tensor.shape).astype(np_dtype)
in_0.tofile("in_0.bin")
in_1.tofile("in_1.bin")
output_concat.tofile("output_concat.bin")
out_np = out_tensor.numpy()
out_np.tofile("out.bin")
output_concat_grad.tofile("output_concat_grad.bin")
feature_0_grad.tofile("in_0_grad.bin")
feature_1_grad.tofile("in_1_grad.bin")
dy.tofile("dy.bin")
