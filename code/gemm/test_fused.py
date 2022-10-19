import oneflow as flow
import oneflow.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(
        self, in_features: int, hidden_units, skip_final_activation=False, fused=True
    ) -> None:
        super(MLP, self).__init__()
        if fused:
            self.linear_layers = nn.FusedMLP(
                in_features,
                hidden_units[:-1],
                hidden_units[-1],
                skip_final_activation=skip_final_activation,
            )
        else:
            units = [in_features] + hidden_units
            num_layers = len(hidden_units)
            denses = [
                Dense(units[i], units[i + 1], not skip_final_activation or (i + 1) < num_layers)
                for i in range(num_layers)
            ]
            self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0.0, np.sqrt(2 / sum(param.shape)))
            elif "bias" in name:
                nn.init.normal_(param, 0.0, np.sqrt(1 / param.shape[0]))

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)

top_mlp=[1024,1024,512,256]
mlp = MLP(
            480,
            top_mlp + [1],
            skip_final_activation=True,
            fused=True,
        )
mlp.to("cuda")
np_x = np.random.uniform(low=-1, high=1, size=(6912, 480)).astype(np.float16)
x = flow.tensor(np_x, device="cuda", dtype=flow.float16, requires_grad=True)

y = mlp(x)
loss = y.sum()
loss.backward()
