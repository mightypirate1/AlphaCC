from typing import Self

from alpha_cc.api.game_manager.db import DBGameState
from alpha_cc.api.io.base_io import BaseIO
from alpha_cc.api.io.board_io import BoardIO


class GameIO(BaseIO):
    game_id: str
    boards: list[BoardIO]

    @classmethod
    def from_db_state(cls: type[Self], game_id: str, db_state: DBGameState) -> Self:
        last_moves = [None, *db_state.moves]
        return cls(
            game_id=game_id,
            boards=[
                BoardIO.from_board(
                    state.board,
                    last_move=last_move,
                )
                for state, last_move in zip(db_state.states, last_moves)
            ],
        )
