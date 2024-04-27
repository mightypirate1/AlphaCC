from typing import Self

from pydantic import BaseModel

from alpha_cc.api.io.move_io import MoveIO
from alpha_cc.engine import Board, Move


class BoardIO(BaseModel):
    game_id: str
    matrix: list[list[int]]
    current_player: int
    game_over: bool
    winner: int
    legal_moves: list[MoveIO]
    last_move: MoveIO | None = None

    @classmethod
    def from_board(cls: type[Self], game_id: str, board: Board, last_move: Move | None = None) -> Self:
        legal_move_ios = [MoveIO.from_move(move, index=i) for i, move in enumerate(board.get_legal_moves())]
        last_move_io = MoveIO.from_move(last_move) if last_move is not None else None
        return cls(
            game_id=game_id,
            matrix=board.get_matrix(),
            current_player=board.board_info.current_player,
            game_over=board.board_info.game_over,
            winner=board.board_info.winner,
            legal_moves=legal_move_ios,
            last_move=last_move_io,
        )
