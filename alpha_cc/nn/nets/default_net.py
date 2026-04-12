from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from alpha_cc.nn.blocks import PolicySoftmax, ResBlock
from alpha_cc.nn.nets.preprocessing import CoordinateChannels

if TYPE_CHECKING:
    from alpha_cc.engine import GameConfig
    from alpha_cc.state import GameState


class DefaultNet(torch.nn.Module):
    def __init__(
        self,
        config: GameConfig,
        n_blocks: int = 6,
        hidden_channels: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        ch = hidden_channels
        board_size = config.board_size
        in_channels = config.state_channels + CoordinateChannels.EXTRA_CHANNELS
        # policy head outputs (board_size**2 channels * board_size * board_size spatial)
        # which reshapes to policy_shape
        policy_channels = config.policy_size // (board_size * board_size)
        self._board_size = board_size
        self._policy_shape = (-1, *config.policy_shape)
        self._policy_size = config.policy_size
        self._preprocessing = CoordinateChannels(config)
        self._encoder = torch.nn.Sequential(
            ResBlock(in_channels, ch, 3),
            *(ResBlock(ch, ch, 5) for _ in range(n_blocks - 1)),
        )
        self._value_local_encoder = torch.nn.Sequential(
            ResBlock(ch, ch, 5),
            torch.nn.AvgPool2d(board_size),
            torch.nn.Flatten(),
        )
        self._value_global_encoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(ch, ch // 2),
            torch.nn.ReLU(),
        )
        self._policy_head = torch.nn.Sequential(
            torch.nn.Dropout2d(dropout),
            ResBlock(ch, ch, 5),
            torch.nn.Conv2d(ch, policy_channels, 5, padding=2),
        )
        self._wdl_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ch + ch // 2, ch // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(ch // 2, 3),  # WDL logits: (win, draw, loss)
        )
        self._policy_softmax = PolicySoftmax(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # preprocessing
        x = self._preprocessing(x)

        # encoder
        x_enc = self._encoder(x)

        # policy head
        x_policy = self._policy_head(x_enc)
        x_pi = x_policy.view(self._policy_shape)

        # value head — WDL logits, shape (n, 3)
        x_local = self._value_local_encoder(x_enc)
        x_global = self._value_global_encoder(x_enc)
        x_wdl = self._wdl_head(torch.cat([x_local, x_global], dim=1))
        return x_pi, x_wdl

    def param_groups(self) -> dict[str, torch.nn.Module]:
        return {
            "encoder": self._encoder,
            "value-local-encoder": self._value_local_encoder,
            "value-global-encoder": self._value_global_encoder,
            "policy-head": self._policy_head,
            "wdl-head": self._wdl_head,
        }

    @torch.no_grad()
    def evaluate_state(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pi_probs, wdl_probs) where wdl_probs is [win, draw, loss]."""
        self.eval()
        board = state.board
        st = torch.as_tensor(board.state_tensor()).reshape(1, -1, self._board_size, self._board_size).float()
        x_pi_all, x_wdl_logits = self(st)
        mask = torch.as_tensor(board.policy_mask().astype(bool)).reshape(self._policy_shape[1:])
        x_pi = self._policy_softmax(x_pi_all, mask)
        # Extract legal-move probabilities via flat policy indices
        pi_flat = x_pi.reshape(1, -1)
        indices = torch.as_tensor(board.policy_indices(), dtype=torch.long)
        x_pi_vec = pi_flat[:, indices]
        wdl_probs = torch.nn.functional.softmax(x_wdl_logits, dim=-1)
        return x_pi_vec.squeeze(0).cpu().numpy(), wdl_probs.squeeze(0).cpu().numpy()
