import logging
import traceback

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from alpha_cc.api.expceptions import ServiceExceptionError
from alpha_cc.api.game_manager.game_manager import GameManager
from alpha_cc.api.io import ApplyMoveIO, GameIO, NewGameIO, RequestMoveIO
from alpha_cc.api.io.mcts_node_io import MCTSNodeIO
from alpha_cc.config import Environment
from alpha_cc.db.games_db import GamesDB

app = FastAPI()
game_manager = GameManager(GamesDB(Environment.host_redis))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__file__)


@app.post("/new-game")
async def new_game(request: NewGameIO) -> GameIO:
    try:
        game_id, db_state = game_manager.create_game(request.size, game_id=request.game_id)
        return GameIO.from_db_state(game_id=game_id, db_state=db_state)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.get("/list-games")
async def list_games() -> list[str]:
    try:
        return game_manager.list_games()
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.get("/fetch-game")
async def fetch_game(game_id: str) -> GameIO:
    try:
        db_state = game_manager.fetch_game(game_id)
        return GameIO.from_db_state(game_id=game_id, db_state=db_state)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.get("/fetch-mcts-node")
async def fetch_mcts_node(game_id: str, board_index: int) -> MCTSNodeIO:
    try:
        node = game_manager.fetch_mcts_node(game_id, board_index)
        return MCTSNodeIO.from_mcts_node(node)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.post("/apply-move")
async def apply_move(request: ApplyMoveIO) -> GameIO:
    try:
        db_state = game_manager.apply_move(request.game_id, request.move_index)
        return GameIO.from_db_state(game_id=request.game_id, db_state=db_state)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.post("/request-move")
async def request_move(request: RequestMoveIO) -> GameIO:
    try:
        db_state = game_manager.request_move(
            request.game_id, request.n_rollouts, request.rollout_depth, request.temperature
        )
        return GameIO.from_db_state(game_id=request.game_id, db_state=db_state)
    except Exception as e:
        raise ServiceExceptionError(e) from e


@app.exception_handler(ServiceExceptionError)
async def handle_service_exception(request: Request, exc: ServiceExceptionError) -> Response:
    logger.error(f"Service exception: {exc} when processing request {request}")
    logger.error(traceback.format_exc())
    return exc.to_response()
