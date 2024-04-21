import torch


class PolicySoftmax(torch.nn.Module):
    def __init__(self, board_size: int) -> None:
        super().__init__()
        self._flat_shape = (-1, board_size**4)
        self._tensor_shape = (-1, board_size, board_size, board_size, board_size)

    def forward(self, x_pi: torch.Tensor, legal_move_mask: torch.BoolTensor) -> torch.Tensor:
        # - mask out illegal moves
        # - flatten
        # - softmax
        # - unflatten
        x_pi = torch.where(legal_move_mask, x_pi, -torch.inf)
        x_pi = x_pi.reshape(self._flat_shape)
        x_pi = torch.nn.functional.softmax(x_pi, dim=1)
        return x_pi.reshape(self._tensor_shape)
