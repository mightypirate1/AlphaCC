from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alpha_cc.api.game_manager.faux_db import FauxDB
from alpha_cc.api.game_manager.game_manager import GameManager
from alpha_cc.api.io import ApplyMoveIO, BoardIO, NewGameIO, RequestMoveIO
from alpha_cc.engine import Board

app = FastAPI()
game_manager = GameManager(FauxDB())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/static-board")
async def static_board() -> BoardIO:
    return BoardIO.from_board(game_id="static", board=Board(9))


@app.post("/new-game")
async def new_game(request: NewGameIO) -> BoardIO:
    game_id, board = game_manager.create_game(request.size, game_id=request.game_id)
    return BoardIO.from_board(game_id=game_id, board=board)


@app.post("/apply-move")
async def apply_move(request: ApplyMoveIO) -> BoardIO:
    move, board = game_manager.apply_move(request.game_id, request.move_index)
    return BoardIO.from_board(game_id=request.game_id, board=board, last_move=move)


@app.post("/request-move")
async def request_move(request: RequestMoveIO) -> BoardIO:
    move, resulting_board = game_manager.request_move(request.game_id, request.time_limit)
    return BoardIO.from_board(game_id=request.game_id, board=resulting_board, last_move=move)
