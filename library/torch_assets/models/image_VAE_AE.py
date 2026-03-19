import torch
import torch.nn as nn
from ..layers import ConvNextLayer, ConvNextStage
from ..layers.conv2d import Conv2d

class ImageEncoder(nn.Module):
    def __init__(self, in_dims: int, num_blocks_per_stage: int, stage_dims: list[int], final_dims: int):
        super().__init__()
        self.encoder = ConvNextLayer(in_dims, num_blocks_per_stage, stage_dims)
        self.final_act = nn.GELU()
        self.final_conv = Conv2d(stage_dims[-1], final_dims*2, 3, 1, 1, True, True)

    def forward(self, x):
        return self.final_conv(self.final_act(self.encoder(x)))


class ImageDecoder(nn.Module):
    def __init__(self, in_dims: int, num_blocks_per_stage: int, stage_dims: list[int], encoded_dims: int):
        super().__init__()
        self.decode_conv = Conv2d(encoded_dims, stage_dims[-1], 3, 1, 1, True, True)
        self.activation = nn.GELU()
        self.up_stages = []
        for i in range(len(stage_dims) - 1, 0, -1):
            self.up_stages.append(
                nn.ConvTranspose2d(stage_dims[i], stage_dims[i - 1], kernel_size=2, stride=2)
            )
            self.up_stages.append(ConvNextStage(num_blocks_per_stage, stage_dims[i - 1]))
        self.up_stages.append(nn.ConvTranspose2d(stage_dims[0], stage_dims[0], kernel_size=4, stride=4))
        self.up_stages.append(nn.GELU())
        self.up_stages.append(nn.Conv2d(stage_dims[0], in_dims, 1, 1))
        self.up_stages.append(nn.Sigmoid())
        self.up_stages = nn.Sequential(*self.up_stages)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up_stages(self.activation(self.decode_conv(z)))


class ImageAE(nn.Module):
    def __init__(self, in_dims: int, num_blocks_per_stage: int, stage_dims: list[int]):
        super().__init__()
        self.encoder = ImageEncoder(in_dims, num_blocks_per_stage, stage_dims)
        self.decoder = ImageDecoder(in_dims, num_blocks_per_stage, stage_dims)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

class ImageVAE(nn.Module):
    def __init__(self, in_dims: int, num_blocks_per_stage: int, stage_dims: list[int], encoded_dims: int):
        super().__init__()
        self.encoder = ImageEncoder(in_dims, num_blocks_per_stage, stage_dims, encoded_dims)
        self.decoder = ImageDecoder(in_dims, num_blocks_per_stage, stage_dims, encoded_dims)
    
    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, 1)
        return mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        x = self.decode(z)
        return x