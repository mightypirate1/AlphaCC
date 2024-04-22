from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alpha_cc.api.io import ApeIO, BoardIO
from alpha_cc.engine import Board

BOARD_MATRIX = Board(9).get_matrix_from_perspective_of_current_player()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/hello-world/{bacon}")
async def hello_world(ape: ApeIO, bacon: int) -> BoardIO:
    return BoardIO(
        message=f"{bacon} hello, {ape}",
        matrix=BOARD_MATRIX,
    )


@app.get("/static-board")
async def static_board() -> BoardIO:
    return BoardIO(
        message="starting board",
        matrix=BOARD_MATRIX,
    )
