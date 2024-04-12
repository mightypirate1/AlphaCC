from pydantic import BaseModel


class BoardIO(BaseModel):
    message: str
    matrix: list[list[int]]
