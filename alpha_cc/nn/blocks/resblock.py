import torch


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
        padding = kernel_size // 2 if conv_pad else 0

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(out_channels) if apply_batch_norm else torch.nn.Identity()
        self.bn2 = torch.nn.BatchNorm2d(out_channels) if apply_batch_norm else torch.nn.Identity()
        self.relu = torch.nn.ReLU()

        if in_channels != out_channels:
            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)
