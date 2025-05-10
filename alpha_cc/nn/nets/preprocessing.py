import torch


class CoordinateChannels(torch.nn.Module):
    """
    Concatenate coordinate channels to the input tensor.
    """

    h_coords: torch.Tensor
    w_coords: torch.Tensor

    def __init__(self, board_size: int) -> None:
        super().__init__()
        h_coords = torch.linspace(0, 1, board_size).view(1, 1, -1, 1)
        w_coords = torch.linspace(0, 1, board_size).view(1, 1, 1, -1)
        self.register_buffer("h_coords", h_coords)
        self.register_buffer("w_coords", w_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h_expanded = self.h_coords.expand(batch_size, 1, -1, self.w_coords.size(3))
        w_expanded = self.w_coords.expand(batch_size, 1, self.h_coords.size(2), -1)
        return torch.cat([x, h_expanded, w_expanded], dim=1)
