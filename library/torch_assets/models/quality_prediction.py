import torch 
import torch.nn as nn 

from ..layers.conv2d import Conv2d
from ..layers import ConvNextStage

class LatentQualityPredictorNetwork(nn.Module):
    def __init__(self, in_dims: int, projection_dims: int, feature_dims: list[int], dense_dims: list[int]):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Sequential(
            Conv2d(in_dims, projection_dims, 3, 1, 1, True, True),
            nn.GELU()
        )
        )
    
        for idx, _feature in enumerate(feature_dims):
            _in_channels = projection_dims if idx == 0 else feature_dims[idx - 1]
            _out_channels = feature_dims[idx]
            self.layers.append(
                nn.Sequential(
                Conv2d(_in_channels, _out_channels, 5, 2, 2, True, True),
                nn.GELU(),
                ConvNextStage(2, _out_channels)
                )
            )

        self.layers.append(nn.AdaptiveAvgPool2d(1))
        self.layers.append(nn.Flatten())
    
        for idx, _dense_dim in enumerate(dense_dims):
            _in_channels = feature_dims[-1] if idx == 0 else dense_dims[idx - 1]
            _out_channels = _dense_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(_in_channels, _out_channels, True),
                    nn.GELU()
                )
            )

        self.layers.append(nn.Linear(dense_dims[-1], 1))
        self.layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*self.layers)



    def forward(self, x):
        x = self.layers(x)
        return x 