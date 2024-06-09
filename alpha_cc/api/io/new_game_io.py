from typing import Literal

from alpha_cc.api.io.base_io import BaseIO


class NewGameIO(BaseIO):
    size: Literal[5, 7, 9]
    game_id: str | None
    show_mode: bool = False
