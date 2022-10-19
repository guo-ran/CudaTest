import oneflow as flow 
import numpy as np 

import torch 

device = "cpu"

np_x = np.array([[[0.5610]],
                 [[0.8279]],
                 [[0.5823]],
                 [[0.5469]]]).astype(np.float32)
np_y = np.array([[[0.7303, 0.2793, 0.3120],
                  [0.6989, 0.9001, 0.6113]]]).astype(np.float32)


x = flow.tensor(np_x, device=device, dtype=flow.float32, requires_grad=True)
y = flow.tensor(np_y, device=device, dtype=flow.float32, requires_grad=True)

out = flow.minimum(x, y)
print(out)
loss = out.sum()
loss.backward()

torch_x = torch.tensor(np_x, device=device, dtype=torch.float32, requires_grad=True)
torch_y = torch.tensor(np_y, device=device, dtype=torch.float32, requires_grad=True)

torch_out = torch.minimum(torch_x, torch_y)
print(torch_out)
torch_loss = torch_out.sum()
torch_loss.backward()

print("Is out equal?", np.allclose(torch_out.detach().cpu().numpy(), out.numpy()))
print("Is x grad equal? ", np.allclose(torch_x.grad.detach().cpu().numpy(), x.grad.numpy()))
print("Is y grad equal? ", np.allclose(torch_y.grad.detach().cpu().numpy(), y.grad.numpy()))
