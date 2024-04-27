import torch
from einops import reduce


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        apply_batch_norm: bool = True,
        conv_pad: bool = True,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        # TODO: figure out why it works this way size-wise (it seems reversed?)
        self._padding = (0, 0, 0, 0, 0, self._out_channels - self._in_channels)
        self._layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2 if conv_pad else 0,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2 if conv_pad else 0,
                ),
                torch.nn.ReLU(),
            ]
        )
        if apply_batch_norm:
            self._layers.append(torch.nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._in_channels > self._out_channels:
            x_res = reduce(
                x,
                "b c_in h w -> b c_out h w",
                "max",
                c_in=self._in_channels,
                c_out=self._out_channels,
            )
        else:
            x_res = torch.nn.functional.pad(x, self._padding, "constant", 0)

        for layer in self._layers:
            x = layer(x)

        return x + x_res
