import numpy as np
import torch
from lru import LRU

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.state import GameState, StateHash
from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.nn.nets.dual_head_net import DualHeadNet


class DefaultNet(torch.nn.Module, DualHeadNet[list[list[MCTSExperience]]]):
    def __init__(self, board_size: int, cache_size: int = 10000) -> None:
        super().__init__()
        self._cache: LRU[StateHash, tuple[torch.Tensor, torch.Tensor]] = LRU(cache_size)
        self._board_size = board_size

        self._encoder = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(1, 64, 3, padding=1),
                torch.nn.Conv2d(64, 128, 5, padding=2),
                torch.nn.Conv2d(128, 128, 5, padding=2),
            ]
        )
        self._policy_head = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(128, board_size * board_size, 5, padding=2),
            ]
        )
        self._value_head = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(128, 128, 7),
                torch.nn.Conv2d(128, 128, 3),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self._encoder:
            x = layer(x)
        x_enc = x
        for layer in self._value_head:
            x = layer(x)

        x_value = x.squeeze(1)  # shape (n,)

        x = x_enc
        for layer in self._policy_head:
            x = layer(x)

        x_pi = x.view(
            -1,
            self._board_size,  # from_x
            self._board_size,  # from_y
            self._board_size,  # to_x
            self._board_size,  # to_y
        )  # shape (n, d, d, d, d)
        return x_pi, x_value

    @torch.no_grad()
    def policy(self, state: GameState) -> np.ndarray:
        x_pi_all, _ = self._create_or_get_cached_output(state)
        # translate nn action to mcts actio
        x_pi = x_pi_all[:, *action_indexer(state.board)]
        return torch.nn.functional.softmax(x_pi, dim=0).squeeze(0).numpy()

    @torch.no_grad()
    def value(self, state: GameState) -> np.floating:
        _, x_value = self._create_or_get_cached_output(state)
        return x_value.squeeze().numpy()

    def clear_cache(self) -> None:
        self._cache.clear()

    @torch.no_grad()
    def _create_or_get_cached_output(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        if state.hash not in self._cache:
            x = (
                torch.as_tensor(state.matrix)
                .reshape(
                    1,
                    1,
                    self._board_size,
                    self._board_size,
                )
                .float()
            )

            self._cache[state.hash] = self(x)
        return self._cache[state.hash]
