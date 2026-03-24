from ..layers import UNetLayer
from .image_VAE_AE import ImageVAE
from ...helper import load_torch_model
import torch 
import torch.nn as nn 

class LatentRefinerNetwork(nn.Module):
    def __init__(self, in_dims: int, projection_dims: int, encode_filters: list[int], conv_next_blocks: int, 
                bottleneck_filters: int, vae_path: str):
        super().__init__()
        self.unet_layer = UNetLayer(in_dims, projection_dims, encode_filters, conv_next_blocks, bottleneck_filters)
        self.image_vae = load_torch_model(vae_path, ImageVAE)
        for param in self.image_vae.parameters():
            param.requires_grad = False
        self.image_vae.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_vae.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            x_enc, _ = self.image_vae.encode(x)
        x_refined = self.unet_layer(x_enc)
        x_dec = self.image_vae.decode(x_refined)
        return x_dec