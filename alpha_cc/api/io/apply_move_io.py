from pydantic import BaseModel


class ApplyMoveIO(BaseModel):
    game_id: str
    move_index: int
