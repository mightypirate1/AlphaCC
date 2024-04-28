from typing import Literal

from pydantic import BaseModel


class NewGameIO(BaseModel):
    size: int
    game_id: str | None
    starting_player: Literal[1, 2] | None
