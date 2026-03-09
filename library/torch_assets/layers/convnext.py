import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalResponseNormalization(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class ConvNextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, padding_mode='replicate')
        self.layer_norm = nn.LayerNorm(dim)
        self.pw_conv_1 = nn.Conv2d(dim, dim*4, 1)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNormalization(dim*4)
        self.pw_conv_2 = nn.Conv2d(dim*4, dim, 1, 1)
    
    def forward(self, x):
        x_res = x 
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pw_conv_1(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.pw_conv_2(x)
        x = x + x_res
        return x 

class ConvNextStage(nn.Module):
    def __init__(self, num_blocks, dim):
        super().__init__()
        self.layers = [] 
        for _ in range(num_blocks):
            self.layers.append(ConvNextBlock(dim))
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)

class ConvNextLayer(nn.Module):
    def __init__(self, in_dim, num_blocks, stage_dims):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_dim, stage_dims[0], 4, 4)
        self.conv_stages = [] 
        for idx in range(len(stage_dims) - 1):
            self.conv_stages.append(ConvNextStage(num_blocks, stage_dims[idx]))
            if idx != (len(stage_dims) - 1):
                self.conv_stages.append(nn.Conv2d(stage_dims[idx], stage_dims[idx+1], 2, 2))
        self.conv_stages = nn.Sequential(*self.conv_stages)
        self.average_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.encode_conv(x)
        x = self.conv_stages(x)
        x = self.average_pool(x)
        x = torch.flatten(x, 1)
        return x 