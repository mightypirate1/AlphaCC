import numpy as np
import torch

from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.nn.blocks import PolicySoftmax, ResBlock
from alpha_cc.state import GameState


class DefaultNet(torch.nn.Module):
    def __init__(self, board_size: int, dropout: float = 0.3) -> None:
        super().__init__()
        self._board_size = board_size

        self._encoder = torch.nn.ModuleList(
            [
                ResBlock(2, 64, 3),
                ResBlock(64, 128, 5),
            ]
        )
        self._policy_head = torch.nn.ModuleList(
            [
                torch.nn.Dropout2d(dropout),
                ResBlock(128, 128, 5),
                torch.nn.Conv2d(128, board_size * board_size, 5, padding=2),
            ]
        )
        self._value_head = torch.nn.ModuleList(
            [
                # ResBlock(128, 128, 5),
                torch.nn.AvgPool2d(board_size),
                torch.nn.Flatten(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 1),
                torch.nn.Tanh(),
            ]
        )
        self._policy_softmax = PolicySoftmax(board_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder
        for layer in self._encoder:
            x = layer(x)
        x_enc = x

        # policy head
        for layer in self._policy_head:
            x = layer(x)
        x_pi = x.view(  # form tensor pi
            -1,
            self._board_size,  # from_x
            self._board_size,  # from_y
            self._board_size,  # to_x
            self._board_size,  # to_y
        )

        # value head
        x = x_enc
        for layer in self._value_head:
            x = layer(x)
        x_value = x.squeeze(1)  # shape (n,)

        return x_pi, x_value

    @torch.no_grad()
    def evaluate_state(self, state: GameState) -> tuple[np.ndarray, np.floating]:
        self.eval()
        x = state.tensor.unsqueeze(0)
        x_pi_all, x_value = self(x)
        mask = torch.as_tensor(state.action_mask)
        x_pi = self._policy_softmax(x_pi_all, mask)
        x_pi_vec = x_pi[:, *action_indexer(state)]
        return x_pi_vec.squeeze(0).numpy(), x_value.squeeze().numpy()
