import torch 
import torch.nn as nn 
from .conv2d import Conv2d

class FiLMBlock(nn.Module):
    def __init__(self, in_dims, feature_dims):
        super().__init__()
        self.proj_1 = nn.Linear(feature_dims, in_dims)
        self.proj_2 = nn.Linear(feature_dims, in_dims)
    
    def forward(self, x_block, x_feature):
        scale = self.proj_1(x_feature)
        bias = self.proj_2(x_feature)
        scale = torch.unsqueeze(torch.unsqueeze(scale, -1), -1)
        bias = torch.unsqueeze(torch.unsqueeze(bias, -1), -1)
        return (x_block * scale) + bias

class PostProcessorBlock(nn.Module):
    def __init__(self, dims, feature_dims):
        super().__init__()
        self.film_layer = FiLMBlock(dims, feature_dims)
        self.conv_1 = Conv2d(dims, dims, 3, 1, 1, True, True)
        self.activation = nn.LeakyReLU()
        self.conv_2 = Conv2d(dims, dims, 3, 1, 1, True, True)

    def forward(self, x_block, x_feature):
        x_res = x_block 
        x = self.film_layer(x_block, x_feature)
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x + x_res 
        return x 

class PostProcessor(nn.Module):
    def __init__(self, in_dims, encode_channels, downsample_amts, bottleneck_dims, num_bottleneck_blocks, enc_feature_dims):
        super().__init__()
        self.encode_layers = []
        self.encode_layers.append(Conv2d(in_dims, encode_channels, 3, 1, 1, True, True))
        self.encode_layers.append(nn.LeakyReLU())
        for i in range(downsample_amts):
            _channels = encode_channels * (4**(i+1))
            print(_channels)
            self.encode_layers.append(nn.PixelUnshuffle(2))
            self.encode_layers.append(Conv2d(_channels, _channels, 3, 1, 1, True, True))
            self.encode_layers.append(nn.LeakyReLU())
        self.encode_layers = nn.Sequential(*self.encode_layers)

        self.bottleneck_broadcast = Conv2d(_channels, bottleneck_dims, 1, 1, 0, True, False)
        self.bottleneck_layers = []
        for _ in range(num_bottleneck_blocks):
            self.bottleneck_layers.append(PostProcessorBlock(bottleneck_dims, enc_feature_dims))
        self.bottleneck_layers = nn.Sequential(*self.bottleneck_layers)
    
        self.decode_broadcast = Conv2d(bottleneck_dims, _channels, 1, 1, 0, True, False)
        self.decode_layers = []
        for i in range(downsample_amts):
            _decode_ch = encode_channels * (4 ** (downsample_amts - i))
            self.decode_layers.append(Conv2d(_decode_ch, _decode_ch, 3, 1, 1, True, True))
            self.decode_layers.append(nn.LeakyReLU())
            self.decode_layers.append(nn.PixelShuffle(2))
        self.decode_layers = nn.Sequential(*self.decode_layers)

        self.final_conv = Conv2d(encode_channels, in_dims, 3, 1, 1, True, False)

    def forward(self, x, enc_features):
        x_res = x 
        x = self.encode_layers(x)
        x = self.bottleneck_broadcast(x)
        for block in self.bottleneck_layers:
            x = block(x, enc_features)
        x = self.decode_broadcast(x)
        x = self.decode_layers(x)
        x = self.final_conv(x)
        x = x + x_res 
        return x
