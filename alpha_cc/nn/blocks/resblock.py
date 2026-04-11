import torch

from alpha_cc.nn.blocks.se_block import SEBlock


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        apply_batch_norm: bool = True,
        conv_pad: bool = True,
        se_reduction: int | None = 16,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if conv_pad else 0

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(out_channels) if apply_batch_norm else torch.nn.Identity()
        self.bn2 = torch.nn.BatchNorm2d(out_channels) if apply_batch_norm else torch.nn.Identity()
        self.se = SEBlock(out_channels, se_reduction) if se_reduction is not None else torch.nn.Identity()
        self.relu = torch.nn.ReLU()
        self.skip = (
            torch.nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return self.relu(x + residual)
