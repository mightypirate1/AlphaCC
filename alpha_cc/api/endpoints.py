import logging
import traceback

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from alpha_cc.api.expceptions import ServiceExceptionError
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


logger = logging.getLogger(__file__)


@app.get("/static-board")
async def static_board() -> BoardIO:
    try:
        return BoardIO.from_board(game_id="static", board=Board(9))
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.post("/new-game")
async def new_game(request: NewGameIO) -> BoardIO:
    try:
        game_id, board = game_manager.create_game(request.size, game_id=request.game_id)
        return BoardIO.from_board(game_id=game_id, board=board)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.post("/apply-move")
async def apply_move(request: ApplyMoveIO) -> BoardIO:
    try:
        applied_move, resulting_board = game_manager.apply_move(request.game_id, request.move_index)
        return BoardIO.from_board(game_id=request.game_id, board=resulting_board, last_move=applied_move)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.post("/request-move")
async def request_move(request: RequestMoveIO) -> BoardIO:
    try:
        applied_move, resulting_board = game_manager.request_move(
            request.game_id, request.n_rollouts, request.rollout_depth, request.temperature
        )
        return BoardIO.from_board(game_id=request.game_id, board=resulting_board, last_move=applied_move)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.exception_handler(ServiceExceptionError)
async def handle_service_exception(request: Request, exc: ServiceExceptionError) -> Response:
    logger.error(f"Service exception: {exc} when processing request {request}")
    logger.error(traceback.format_exc())
    return exc.to_response()
