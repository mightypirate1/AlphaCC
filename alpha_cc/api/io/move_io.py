from typing import Self

from alpha_cc.api.io.base_io import BaseIO
from alpha_cc.api.io.coord_io import CoordIO
from alpha_cc.engine import Board, Move


class MoveIO(BaseIO):
    from_coord: CoordIO
    to_coord: CoordIO
    path: list[CoordIO]
    index: int | None

    @classmethod
    def from_move(cls: type[Self], move: Move, board: Board, index: int | None = None) -> Self:
        """
        "Flips" moves since they are created from a potentially flipped board in the engine.
        """
        from_coord = CoordIO(
            x=move.from_coord.x if board.info.current_player == 1 else board.info.size - 1 - move.from_coord.x,
            y=move.from_coord.y if board.info.current_player == 1 else board.info.size - 1 - move.from_coord.y,
        )
        to_coord = CoordIO(
            x=move.to_coord.x if board.info.current_player == 1 else board.info.size - 1 - move.to_coord.x,
            y=move.to_coord.y if board.info.current_player == 1 else board.info.size - 1 - move.to_coord.y,
        )
        return cls(index=index, from_coord=from_coord, to_coord=to_coord, path=[from_coord, to_coord])
