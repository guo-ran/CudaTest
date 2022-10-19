import oneflow as flow
import oneflow.nn as nn 
import numpy as np 

flow_dtype = flow.float32


x = flow.tensor(np.random.randn(13824, 480).astype(np.float32), dtype=flow_dtype, device="cuda", requires_grad=True)
fused_mlp = nn.FusedMLP(480, [400, 400, 400, 400, 400], out_features=1, skip_final_activation=False).to("cuda")


out = fused_mlp(x)
out_sum = out.sum()
out_sum.backward()
print(x.grad)
