import oneflow as flow
import numpy as np

batch_size = 55296//8
embedding_size = 128
#in_0 = np.random.rand(batch_size, 1, embedding_size).astype(np.float16)
#in_1 = np.random.rand(batch_size, 26, embedding_size).astype(np.float16)
in_0 = np.fromfile("in_0.bin", np.float16).reshape(batch_size, 1, embedding_size)
in_1 = np.fromfile("in_1.bin", np.float16).reshape(batch_size, 26, embedding_size)
print("in_0",in_0[0])
print("in_1",in_1[0])
in_0_tensor = flow.tensor(in_0, device="cuda")
in_1_tensor = flow.tensor(in_1, device="cuda")
out_tensor = flow._C.fused_dot_feature_interaction([in_0_tensor, in_1_tensor], 
        output_concat=flow.reshape(in_0_tensor, (batch_size, embedding_size)),
        self_interaction=False,
        output_padding=1,
        pooling="none")
print("out",out_tensor)
#in_0.tofile("in_0.bin")
#in_1.tofile("in_1.bin")
#out_np = out_tensor.numpy()
#out_np.tofile("out.bin")
