from time import sleep
from uuid import uuid4

from alpha_cc.agents import MCTSAgent
from alpha_cc.api.game_manager.db import DB, DBGameState
from alpha_cc.engine import Board, Move
from alpha_cc.state import GameState


class GameManager:
    def __init__(self, db: DB) -> None:
        self._db = db
        self._agents = {size: MCTSAgent(size, n_rollouts=100, rollout_depth=10) for size in self.supported_sizes}

    @property
    def supported_sizes(self) -> list[int]:
        return list(range(100))

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

    def request_move(self, game_id: str, time_limit: float) -> tuple[Move, Board]:
        db_state = self._db.get_entry(game_id)
        agent = self._agents[db_state.state.info.size]
        sleep(time_limit)  # simulate delay
        move_index = agent.choose_move(db_state.state.board)
        move = db_state.state.board.get_moves()[move_index]
        resulting_state = db_state.state.children[move_index]
        self._db.set_entry(game_id, db_state)
        return move, resulting_state.board
