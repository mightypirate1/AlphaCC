from typing import Self

from pydantic import BaseModel

from alpha_cc.api.io.coord_io import CoordIO
from alpha_cc.engine import Move


class MoveIO(BaseModel):
    from_coord: CoordIO
    to_coord: CoordIO
    index: int | None

    @classmethod
    def from_move(cls: type[Self], move: Move, index: int | None = None) -> Self:
        return cls(
            index=index,
            from_coord=CoordIO(x=move.from_coord.x, y=move.from_coord.y),
            to_coord=CoordIO(x=move.to_coord.x, y=move.to_coord.y),
        )
