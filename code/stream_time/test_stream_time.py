import oneflow as flow
import numpy as np
import time

class TestGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
    
    def build(self, input):
        out = input
        for i in range(100):
            out = flow._C.relu(out)
            out = flow._C.leaky_relu(out, alpha=0.1)
        return out
        
graph = TestGraph()

np_in = np.random.rand(25*1024*1024).astype(np.float16)
input = flow.tensor(np_in, device="cuda")

print(graph(input).numpy().flatten()[0:10])
start = time.time()
out = input
for i in range(20):
    out = graph(out)
print(out.numpy().flatten()[0:10])
end = time.time()
print("time ", end-start)

