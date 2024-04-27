from __future__ import annotations

import hashlib
from typing import NewType

from alpha_cc.engine import Board, BoardInfo, HexCoordinate

StateHash = NewType("StateHash", bytes)


class GameState:
    def __init__(self, board: Board, disallowed_states: set[StateHash] | None = None) -> None:
        self._board = board
        self._disallowed_states = disallowed_states
        self._info = board.board_info
        self._matrix: list[list[int]] | None = None
        self._action_mask: list[list[list[list[int]]]] | None = None
        self._action_mask_indices: dict[int, tuple[HexCoordinate, HexCoordinate]] | None = None
        self._children: list[GameState] | None = None
        self._hash: StateHash | None = None

    def __len__(self) -> int:
        return len(self.board.get_all_possible_next_states())

    @property
    def board(self) -> Board:
        return self._board

    @property
    def info(self) -> BoardInfo:
        return self._info

    @property
    def children(self) -> list[GameState]:
        if self._children is None:
            self._children = [GameState(sp) for sp in self.board.get_all_possible_next_states()]
        return self._children

    @property
    def matrix(self) -> list[list[int]]:
        if self._matrix is None:
            self._matrix = self.board.get_matrix_from_perspective_of_current_player()
        return self._matrix

    @property
    def action_mask(self) -> list[list[list[list[int]]]]:
        if self._action_mask is None:
            self._action_mask = self.board.get_legal_moves().get_action_mask()
        return self._action_mask

    @property
    def action_mask_indices(self) -> dict[int, tuple[HexCoordinate, HexCoordinate]]:
        if self._action_mask_indices is None:
            self._action_mask_indices = self.board.get_legal_moves().get_action_mask_indices()
        return self._action_mask_indices

    @property
    def hash(self) -> StateHash:
        if self._hash is None:
            hash_bytes = hashlib.sha256(str(self.matrix).encode()).digest()
            self._hash = StateHash(hash_bytes)
        return self._hash
