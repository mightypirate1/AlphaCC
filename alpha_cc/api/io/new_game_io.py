from typing import Literal

from pydantic import BaseModel


class NewGameIO(BaseModel):
    size: Literal[5, 7, 9]
    game_id: str | None
