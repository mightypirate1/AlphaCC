from pathlib import Path
from time import sleep
from uuid import uuid4

from alpha_cc.agents import StandaloneMCTSAgent
from alpha_cc.api.game_manager.db import DB, DBGameState
from alpha_cc.engine import Board, Move
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.state import GameState


def get_agent(size: int) -> StandaloneMCTSAgent:
    weight_dict = {
        5: Path(__file__).parents[3] / "data/models/test-00-size-5.pth",
    }
    model = StandaloneMCTSAgent(DefaultNet(size), n_rollouts=100, rollout_depth=100)
    if size in weight_dict:
        return model.with_weights(weight_dict[size])
    return model


class GameManager:
    def __init__(self, db: DB) -> None:
        self._db = db
        self._agents = {size: get_agent(size) for size in self.supported_sizes}

    @property
    def supported_sizes(self) -> list[int]:
        return [5, 7, 9]

    def create_game(
        self,
        size: int,
        game_id: str | None = None,
    ) -> tuple[str, Board]:
        if size not in self.supported_sizes:
            raise ValueError(f"size={size} is not supported")
        if game_id in self._db.list_entries():
            raise ValueError(f"game_id={game_id} already exists")
        if game_id is None:
            game_id = str(uuid4())
        board = Board(size)
        db_state = DBGameState(
            state=GameState(board),
            nodes=[],
        )
        self._db.set_entry(game_id, db_state)
        return game_id, board

    def delete_game(self, game_id: str) -> bool:
        return self._db.remove_entry(game_id)

    def apply_move(self, game_id: str, move_index: int) -> tuple[Move, Board]:
        db_state = self._db.get_entry(game_id)
        move = db_state.state.board.get_moves()[move_index]
        resulting_state = db_state.state.children[move_index]
        db_state.state = resulting_state
        self._db.set_entry(game_id, db_state)
        return move, resulting_state.board

    def request_move(self, game_id: str, n_rollouts: int, rollout_depth: int, temperature: float) -> tuple[Move, Board]:
        db_state = self._db.get_entry(game_id)
        agent = self._agents[db_state.state.info.size]
        sleep(1)  # simulate delay
        move_index = agent.choose_move(
            db_state.state.board,
            rollout_depth=rollout_depth,
            n_rollouts=n_rollouts,
            temperature=temperature,
        )
        move = db_state.state.board.get_moves()[move_index]
        resulting_state = db_state.state.children[move_index]
        self._db.set_entry(game_id, db_state)
        return move, resulting_state.board
