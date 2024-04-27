from pydantic import BaseModel


class CoordIO(BaseModel):
    x: int
    y: int
