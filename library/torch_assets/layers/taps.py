import torch
import torch.nn as nn


class TapLayer(nn.Module):
    """MLP tap: linear layers with GELU, optional final activation (e.g. tanh)."""

    def __init__(
        self,
        in_dim: int,
        dense_dims: list[int],
        final_act: str = "tanh",
    ) -> None:
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
                elif final_act == 'GELU':
                    self.dense_layers.append(
                        nn.GELU()
                    )
            else:
                self.dense_layers.append(
                    nn.GELU()
                )
        self.dense_layers = nn.Sequential(*self.dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input vector through the MLP."""
        x = self.dense_layers(x)
        return x