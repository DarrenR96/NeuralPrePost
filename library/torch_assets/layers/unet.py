import torch.nn as nn 
import torch 
from .conv2d import Conv2d
from .convnext import ConvNextStage

class UNetLayer(nn.Module):
    def __init__(self, in_dims: int, projection_dims: int, encode_filters: list[int], conv_next_blocks: int, 
                bottleneck_filters: int):
        super().__init__()
        self.projection_conv = nn.Sequential(
            Conv2d(in_dims, projection_dims, 3, 1, 1, True, True),
            nn.GELU()
        )
        self.encoder_branch = []
        for _idx, _filter in enumerate(encode_filters):
            in_channels = projection_dims if _idx == 0 else (encode_filters[_idx-1])
            out_channels = _filter
            self.encoder_branch.append(
                nn.Sequential(
                    Conv2d(in_channels, out_channels, 5, 2, 2, True, True),
                    nn.GELU(),
                    ConvNextStage(conv_next_blocks, out_channels)
                )
            )
        self.encoder_branch = nn.ModuleList(self.encoder_branch)

        self.bottleneck_branch = []
        self.bottleneck_branch.append(Conv2d(encode_filters[-1], bottleneck_filters, 1, 1, 0, True, True))
        self.bottleneck_branch.append(nn.GELU())
        self.bottleneck_branch.append(ConvNextStage(1, bottleneck_filters))
        self.bottleneck_branch = nn.Sequential(*self.bottleneck_branch)

        decode_skip_channels = list(reversed(encode_filters[:-1])) + [projection_dims]
        decode_out_channels = decode_skip_channels

        self.decoder_branch = []
        self.upsample_branch = []
        decoder_in_channels = [bottleneck_filters] + decode_out_channels[:-1]
        for in_ch, skip_ch, out_ch in zip(decoder_in_channels, decode_skip_channels, decode_out_channels):
            self.upsample_branch.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
            )
            self.decoder_branch.append(
                nn.Sequential(
                    Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, True, True),
                    nn.GELU(),
                    ConvNextStage(conv_next_blocks, out_ch)
                )
            )
        self.upsample_branch = nn.ModuleList(self.upsample_branch)
        self.decoder_branch = nn.ModuleList(self.decoder_branch)

        self.output_head = Conv2d(projection_dims, in_dims, 3, 1, 1, True, True)


    def forward(self, x):
        x = self.projection_conv(x)
        projection_skip = x
        ds_branch_outputs = [] 
        for _layer in self.encoder_branch:
            x = _layer(x)
            ds_branch_outputs.append(x)

        x = self.bottleneck_branch(x)

        skip_tensors = list(reversed(ds_branch_outputs[:-1])) + [projection_skip]
        for upsample_layer, decode_layer, skip in zip(self.upsample_branch, self.decoder_branch, skip_tensors):
            x = upsample_layer(x)
            x = torch.cat([x, skip], dim=1)
            x = decode_layer(x)

        x = self.output_head(x)
        return x