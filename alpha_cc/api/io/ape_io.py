from pydantic import BaseModel


class ApeIO(BaseModel):
    x: str
    y: int
