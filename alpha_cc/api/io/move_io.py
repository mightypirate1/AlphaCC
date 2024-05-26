from typing import Self

from alpha_cc.api.io.base_io import BaseIO
from alpha_cc.api.io.coord_io import CoordIO
from alpha_cc.engine import Move


class MoveIO(BaseIO):
    from_coord: CoordIO
    to_coord: CoordIO
    path: list[CoordIO]
    index: int | None

    @classmethod
    def from_move(cls: type[Self], move: Move, current_player: int, index: int | None = None) -> Self:
        """
        "Flips" moves since they are created from a potentially flipped board in the engine.
        """
        from_coord = move.from_coord if current_player == 1 else move.from_coord.flip()
        to_coord = move.to_coord if current_player == 1 else move.to_coord.flip()
        from_coord_io = CoordIO(x=from_coord.x, y=from_coord.y)
        to_coord_io = CoordIO(x=to_coord.x, y=to_coord.y)
        return cls(index=index, from_coord=from_coord_io, to_coord=to_coord_io, path=[from_coord_io, to_coord_io])
