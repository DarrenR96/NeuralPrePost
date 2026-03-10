import torch
import torch.nn as nn

from ..layers import ConvNextLayer, TapLayer, PostProcessor


class TRINIModel(nn.Module):
    def __init__(
        self,
        encoder_args: dict,
        tap_enc_args: dict,
        tap_dec_args: dict,
        post_processor_args: dict,
    ) -> None:
        super().__init__()
        self.encoder_layer = ConvNextLayer(**encoder_args)
        self.encoder_tap = TapLayer(**tap_enc_args)
        self.decoder_tap = TapLayer(**tap_dec_args)
        self.post_processor = PostProcessor(**post_processor_args)

    def forward(self, x_ref: torch.Tensor, x_comp: torch.Tensor) -> torch.Tensor:
        x_enc = torch.cat([x_ref, x_comp], 1)
        x_enc = self.encoder_layer(x_enc)
        x_enc = self.decoder_tap(self.encoder_tap(x_enc))
        x = self.post_processor(x_comp, x_enc)
        return x 