from alpha_cc.api.io.base_io import BaseIO


class ApplyMoveIO(BaseIO):
    game_id: str
    move_index: int
