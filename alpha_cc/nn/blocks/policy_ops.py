from abc import abstractmethod

import torch


class PolicyOp(torch.nn.Module):
    def __init__(self, board_size: int) -> None:
        super().__init__()
        self._flat_shape = (-1, board_size**4)
        self._tensor_shape = (-1, board_size, board_size, board_size, board_size)

    def forward(self, x_pi_unmasked: torch.Tensor, legal_move_mask: torch.BoolTensor) -> torch.Tensor:
        # - mask out illegal moves
        # - flatten
        # - softmax
        # - unflatten
        x_pi = torch.where(legal_move_mask, x_pi_unmasked, -torch.inf)
        x_pi_flat = x_pi.reshape(self._flat_shape)
        x_pi_flat_oped = self._op_on_flattened_policy(x_pi_flat)
        x_pi_oped_unflattened = x_pi_flat_oped.reshape(self._tensor_shape)
        return x_pi_oped_unflattened

    @abstractmethod
    def _op_on_flattened_policy(self, x_pi_flat: torch.Tensor) -> torch.Tensor:
        pass


class PolicySoftmax(PolicyOp):
    def _op_on_flattened_policy(self, x_pi_flat: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x_pi_flat, dim=1)


class PolicyLogSoftmax(PolicyOp):
    def _op_on_flattened_policy(self, x_pi_flat: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(x_pi_flat, dim=1)
