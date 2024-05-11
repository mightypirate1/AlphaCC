from __future__ import annotations

import hashlib
from typing import NewType, Self

import numpy as np
import torch

from alpha_cc.engine import Board, BoardInfo, HexCoord, Move, create_move_index_map, create_move_mask

StateHash = NewType("StateHash", bytes)


class GameState:
    """
    Wrapper for a `Board` that manages common manipulations on it.
    """

    def __init__(self, board: Board) -> None:
        self._board = board
        self._matrix: np.ndarray | None = None
        self._tensor: torch.Tensor | None = None
        self._action_mask: np.ndarray | None = None
        self._action_mask_indices: dict[int, tuple[HexCoord, HexCoord]] | None = None
        self._children: list[GameState] | None = None
        self._moves: list[Move] | None = None
        self._hash: StateHash | None = None

    def __len__(self) -> int:
        return len(self.board.get_next_states())

    @property
    def board(self) -> Board:
        return self._board

    @property
    def info(self) -> BoardInfo:
        return self.board.info

    @property
    def moves(self) -> list[Move]:
        if self._moves is None:
            self._moves = self.board.get_moves()
        return self._moves

    @property
    def children(self) -> list[GameState]:
        if self._children is None:
            self._children = [GameState(self.board.apply(move)) for move in self.moves]
        return self._children

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            d = self.board.info.size
            self._matrix = np.array(self.board.get_matrix())[:d, :d]
        return self._matrix

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            self._tensor = torch.stack([torch.as_tensor(self.matrix == 1), torch.as_tensor(self.matrix == 2)]).float()
        return self._tensor

    @property
    def action_mask(self) -> np.ndarray:
        if self._action_mask is None:
            moves = self.board.get_moves()
            mask_uncropped = np.array(create_move_mask(moves))
            d = self.board.info.size
            action_mask = mask_uncropped[:d, :d, :d, :d]
            self._action_mask = action_mask
        return self._action_mask

    @property
    def action_mask_indices(self) -> dict[int, tuple[HexCoord, HexCoord]]:
        if self._action_mask_indices is None:
            moves = self.board.get_moves()
            self._action_mask_indices = create_move_index_map(moves)
        return self._action_mask_indices

    @property
    def hash(self) -> StateHash:
        if self._hash is None:
            hash_bytes = hashlib.sha256(self.matrix.tobytes()).digest()
            self._hash = StateHash(hash_bytes)
        return self._hash
    
    def apply(self, move: Move) -> Self:
        return GameState(self.board.apply(move))

    def __getstate__(self) -> object:
        return self._board

    def __setstate__(self, board: Board) -> None:
        self._board = board
        self._matrix = None
        self._tensor = None
        self._action_mask = None
        self._action_mask_indices = None
        self._children = None
        self._moves = None
        self._hash = None
