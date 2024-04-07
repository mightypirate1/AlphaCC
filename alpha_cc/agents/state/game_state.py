from __future__ import annotations

import hashlib
from typing import NewType

from alpha_cc.engine import Board, BoardInfo

StateHash = NewType("StateHash", bytes)


class GameState:
    def __init__(self, board: Board) -> None:
        self._board = board
        self._info = board.board_info
        self._matrix: list[list[int]] | None = None
        self._hash: StateHash | None = None

    @property
    def children(self) -> list[GameState]:
        return [GameState(sp) for sp in self.board.get_all_possible_next_states()]

    @property
    def board(self) -> Board:
        return self._board

    @property
    def info(self) -> BoardInfo:
        return self._info

    @property
    def matrix(self) -> list[list[int]]:
        if self._matrix is None:
            self._matrix = self.board.get_matrix_from_perspective_of_current_player()
        return self._matrix

    @property
    def hash(self) -> StateHash:
        if self._hash is None:
            self._hash = StateHash(
                hashlib.sha256(
                    "".join(
                        [
                            str(self.matrix),
                            str(self.info.current_player),
                            str(self.info.game_over),
                            str(self.info.winner),
                        ]
                    ).encode()
                ).digest()
            )
        return self._hash
