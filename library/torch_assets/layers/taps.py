import torch
import torch.nn as nn
from .conv2d import Conv2d

class TapLayer(nn.Module):
    """Tap: linear layers with GELU, optional final activation (e.g. tanh)."""

    def __init__(
        self,
        in_dim: int,
        cnn_dims: list[int],
        final_act: str = "tanh",
    ) -> None:
        super().__init__()
        self.cnn_layers = []
        self.cnn_layers.extend(
            [
                Conv2d(in_dim, cnn_dims[0], 3, 1, 1, True, True),
                nn.GELU()
            ]
        )
        for idx in range(len(cnn_dims) - 1):
            self.cnn_layers.append(
                Conv2d(cnn_dims[idx], cnn_dims[idx + 1], 3, 1, 1, True, True),
            )
            if idx == len(cnn_dims) - 2:
                if final_act == 'tanh':
                    self.cnn_layers.append(
                        nn.Tanh()
                    )
                elif final_act == 'GELU':
                    self.cnn_layers.append(
                        nn.GELU()
                    )
            else:
                self.cnn_layers.append(
                    nn.GELU()
                )
        self.cnn_layers = nn.Sequential(*self.cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input vector through the MLP."""
        x = self.cnn_layers(x)
        return x