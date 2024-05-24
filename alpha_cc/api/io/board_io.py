from typing import Self

from alpha_cc.api.io.base_io import BaseIO
from alpha_cc.api.io.move_io import MoveIO
from alpha_cc.engine import Board, Move


class BoardIO(BaseIO):
    matrix: list[list[int]]
    current_player: int
    game_over: bool
    evaluation: float
    winner: int
    legal_moves: list[MoveIO]
    last_move: MoveIO | None = None

    @classmethod
    def from_board(cls: type[Self], board: Board, last_move: Move | None = None) -> Self:
        last_player = 3 - board.info.current_player
        legal_move_ios = [
            MoveIO.from_move(move, board.info.current_player, index=i) for i, move in enumerate(board.get_moves())
        ]
        last_move_io = MoveIO.from_move(last_move, last_player) if last_move is not None else None
        return cls(
            matrix=board.get_unflipped_matrix(),
            current_player=board.info.current_player,
            game_over=board.info.game_over,
            evaluation=board.info.reward if board.info.current_player == 1 else -board.info.reward,
            winner=board.info.winner,
            legal_moves=legal_move_ios,
            last_move=last_move_io,
        )
