import torch 
import torch.nn as nn

class Conv2d(nn.Module):
    """2D convolution layer with optional depthwise separable convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all sides of the input.
        bias: If True, adds a learnable bias to the output.
        separable: If True, uses depthwise separable convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, separable):
        super().__init__()
        if separable:
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=in_channels, bias=False,
                padding_mode='replicate'
            )
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=bias
            )
            self.conv_layer = nn.Sequential(self.depthwise, self.pointwise)
        else:
            self.conv_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, padding_mode='replicate')

    def forward(self, x):
        x = self.conv_layer(x)
        return x