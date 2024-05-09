from pydantic import BaseModel


class RequestMoveIO(BaseModel):
    game_id: str
