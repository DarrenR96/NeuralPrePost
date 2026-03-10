import torch 
import torch.nn as nn 

class TapLayer(nn.Module):
    def __init__(self, in_dim, dense_dims, final_act='tanh'):
        super().__init__()
        self.dense_layers = []
        self.dense_layers.extend(
            [
                nn.Linear(in_dim, dense_dims[0]),
                nn.GELU()
            ]
        )
        for idx in range(len(dense_dims) - 1):
            self.dense_layers.append(
                nn.Linear(dense_dims[idx], dense_dims[idx + 1])
            )
            if idx == len(dense_dims) - 2:
                if final_act == 'tanh':
                    self.dense_layers.append(
                        nn.Tanh()
                    )
                else:
                    self.dense_layers.append(
                        nn.GELU()
                    )
            else:
                self.dense_layers.append(
                    nn.GELU()
                )
        self.dense_layers = nn.Sequential(*self.dense_layers)

    def forward(self, x):
        x = self.dense_layers(x)
        return x 