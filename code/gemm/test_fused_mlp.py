import oneflow as flow 
import numpy as np 


np.random.seed(0)

np_x = np.random.uniform(low=-1, high=1, size=(6912, 480)).astype(np.float16)
np_w0 = np.random.uniform(low=-1, high=1, size=(1024, 480)).astype(np.float16)
np_b0 = np.random.uniform(low=-1, high=1, size=(1024,)).astype(np.float16)
np_w1 = np.random.uniform(low=-1, high=1, size=(1024, 1024)).astype(np.float16)
np_b1 = np.random.uniform(low=-1, high=1, size=(1024,)).astype(np.float16)
np_w2 = np.random.uniform(low=-1, high=1, size=(512, 1024)).astype(np.float16)
np_b2 = np.random.uniform(low=-1, high=1, size=(512,)).astype(np.float16)
np_w3 = np.random.uniform(low=-1, high=1, size=(256, 512)).astype(np.float16)
np_b3 = np.random.uniform(low=-1, high=1, size=(256,)).astype(np.float16)
np_w4 = np.random.uniform(low=-1, high=1, size=(1, 256)).astype(np.float16)
np_b4 = np.random.uniform(low=-1, high=1, size=(1,)).astype(np.float16)

gen = flow.Generator("cuda")
gen.manual_seed(0)
flow_type = flow.float16
device="cuda"
requires_grad = True
dropout_rate_list = [0.0] * 3


x = flow.tensor(np_x, device=device, dtype=flow_type, requires_grad=requires_grad)
weight0 = flow.tensor(np_w0, device=device, dtype=flow_type, requires_grad=requires_grad)
bias0 = flow.tensor(np_b0, device=device, dtype=flow_type, requires_grad=requires_grad)
weight1 = flow.tensor(np_w1, device=device, dtype=flow_type, requires_grad=requires_grad)
bias1 = flow.tensor(np_b1, device=device, dtype=flow_type, requires_grad=requires_grad)
weight2 = flow.tensor(np_w2, device=device, dtype=flow_type, requires_grad=requires_grad)
bias2 = flow.tensor(np_b2, device=device, dtype=flow_type, requires_grad=requires_grad)
weight3 = flow.tensor(np_w3, device=device, dtype=flow_type, requires_grad=requires_grad)
bias3 = flow.tensor(np_b3, device=device, dtype=flow_type, requires_grad=requires_grad)
weight4 = flow.tensor(np_w4, device=device, dtype=flow_type, requires_grad=requires_grad)
bias4 = flow.tensor(np_b4, device=device, dtype=flow_type, requires_grad=requires_grad)

for i in range(1):
    fused_out = flow._C.fused_mlp(x, [weight0, weight1, weight2, weight3], [bias0, bias1, bias2, bias3], skip_final_activation=True)
    loss = fused_out.sum()
    loss.backward()
