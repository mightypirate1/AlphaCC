import numpy as np
import torch

from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.nn.blocks import PolicySoftmax, ResBlock
from alpha_cc.nn.nets.preprocessing import CoordinateChannels
from alpha_cc.state import GameState
from alpha_cc.state.state_tensors import state_tensor


class DefaultNet(torch.nn.Module):
    def __init__(
        self,
        board_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._board_size = board_size
        self._preprocessing = CoordinateChannels(board_size)
        self._encoder = torch.nn.Sequential(
            ResBlock(4, 64, 3),  # 4 channels: two from the state tensor, two that we append here
            ResBlock(64, 128, 5),
            ResBlock(128, 128, 5),
        )
        self._global_encoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self._local_encoder = torch.nn.Sequential(
            ResBlock(128, 128, 5),
            torch.nn.AvgPool2d(board_size),
            torch.nn.Flatten(),
        )
        self._policy_head = torch.nn.Sequential(
            torch.nn.Dropout2d(dropout),
            ResBlock(128, 128, 5),
            torch.nn.Conv2d(128, board_size * board_size, 5, padding=2),
        )
        self._value_combined = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128 + 64, 64),  # 128 local + 64 global features
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )
        self._policy_softmax = PolicySoftmax(board_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # preprocessing
        x = self._preprocessing(x)

        # encoder
        x_enc = self._encoder(x)
        x_enc_local = self._local_encoder(x_enc)
        x_enc_global = self._global_encoder(x_enc)
        x_enc_combined = torch.cat([x_enc_local, x_enc_global], dim=1)

        # policy head
        x_policy = self._policy_head(x_enc)
        x_pi = x_policy.view(  # form tensor pi
            -1,
            self._board_size,  # from_x
            self._board_size,  # from_y
            self._board_size,  # to_x
            self._board_size,  # to_y
        )

        # value head
        x_value = self._value_combined(x_enc_combined).squeeze(1)  # shape (n,)
        return x_pi, x_value

    @torch.no_grad()
    def evaluate_state(self, state: GameState) -> tuple[np.ndarray, np.floating]:
        self.eval()
        x = state_tensor(state).unsqueeze(0)
        x_pi_all, x_value = self(x)
        mask = torch.as_tensor(state.action_mask)
        x_pi = self._policy_softmax(x_pi_all, mask)
        x_pi_vec = x_pi[:, *action_indexer(state)]
        return x_pi_vec.squeeze(0).cpu().numpy(), x_value.squeeze().cpu().numpy()
