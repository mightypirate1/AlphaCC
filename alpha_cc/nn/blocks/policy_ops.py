from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from alpha_cc.engine import GameConfig


class PolicyOp(torch.nn.Module):
    def __init__(self, config: GameConfig) -> None:
        super().__init__()
        self._flat_shape = (-1, config.policy_size)
        self._tensor_shape = (-1, *config.policy_shape)

    def forward(self, x_pi_unmasked: torch.Tensor, legal_move_mask: torch.BoolTensor) -> torch.Tensor:
        # - flatten both policy and mask
        # - mask out illegal moves
        # - softmax
        # - unflatten
        x_pi_flat = x_pi_unmasked.reshape(self._flat_shape)
        mask_flat = legal_move_mask.reshape(self._flat_shape)
        x_pi_masked = torch.where(mask_flat, x_pi_flat, -torch.inf)
        x_pi_flat_oped = self._op_on_flattened_policy(x_pi_masked)
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
