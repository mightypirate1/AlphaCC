import logging
from pathlib import Path
from uuid import uuid4

from alpha_cc.agents import StandaloneMCTSAgent
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.node_store import DBNodeStore
from alpha_cc.api.game_manager.show_mode_job import ShowModeJob
from alpha_cc.config import Environment
from alpha_cc.db.games_db import GamesDB
from alpha_cc.db.models import DBGameState
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.state import GameState

WEIGHT_DIR = Path(Environment.model_dir)

logger = logging.getLogger(__name__)


def get_agent(size: int) -> StandaloneMCTSAgent:
    weight_dict = {
        5: WEIGHT_DIR / "api/test-00-size-5.pth",
        7: WEIGHT_DIR / "api/test-00-size-7.pth",
        9: WEIGHT_DIR / "api/test-00-size-9.pth",
    }
    model = StandaloneMCTSAgent(
        DefaultNet(size),
        n_rollouts=500,
        rollout_depth=100,
        rollout_gamma=0.99,
    )
    if size in weight_dict:
        return model.with_weights(weight_dict[size])
    return model


class GameManager:
    def __init__(self, games_db: GamesDB) -> None:
        self._games_db = games_db
        self._agents = {size: get_agent(size) for size in self.supported_sizes}
        self._games_db.clear_show_mode_boards()
        self._show_mode_jobs: dict[str, ShowModeJob] = {}

    @property
    def supported_sizes(self) -> list[int]:
        return [5, 7, 9]

    def create_game(self, size: int, game_id: str | None = None, show_mode: bool = False) -> tuple[str, DBGameState]:
        if size not in self.supported_sizes:
            raise ValueError(f"size={size} is not supported")
        if game_id in self._games_db.list_entries():
            raise ValueError(f"game_id={game_id} already exists")
        if game_id is None:
            game_id = str(uuid4())
        db_state = self._games_db.create_game(game_id, size)

        if show_mode:
            self.show_mode_on(game_id)
        return game_id, db_state

    def fetch_game(self, game_id: str) -> DBGameState:
        return self._games_db.get_state(game_id)

    def fetch_mcts_node(self, game_id: str, board_index: int, as_sorted: bool) -> MCTSNodePy:
        node_store = DBNodeStore(game_id, self._games_db.db)
        db_state = self.fetch_game(game_id)
        state = GameState(db_state.boards[board_index])
        node = node_store.get(state.board)
        if state.info.current_player == 2:
            node = node.with_flipped_value()
        if as_sorted:
            node = node.as_sorted()
        return node

    def list_games(self) -> list[str]:
        return self._games_db.list_entries()

    def delete_game(self, game_id: str) -> bool:
        return self._games_db.remove_entry(game_id)

    def apply_move(self, game_id: str, move_index: int) -> DBGameState:
        logger.info(f"applied move: {move_index} for game {game_id}")
        self._games_db.add_move(game_id, move_index)
        return self._games_db.get_state(game_id)

    def request_move(self, game_id: str, n_rollouts: int, rollout_depth: int, temperature: float) -> DBGameState:
        if game_id in self._show_mode_jobs:
            return self._show_mode_jobs[game_id].request_move()
        db_state = self._games_db.get_state(game_id)
        state = GameState(db_state.boards[-1])
        agent = self._agents[state.info.size]
        db_node_store = DBNodeStore(game_id, self._games_db.db)
        agent.node_store.load_from(db_node_store)
        move_index = agent.choose_move(
            state.board,
            rollout_depth=rollout_depth,
            n_rollouts=n_rollouts,
            temperature=temperature,
        )
        db_node_store.load_from(agent.node_store.fetch_updated())
        db_state.add_move(move_index)
        self._games_db.add_move(game_id, move_index)
        agent.node_store.clear()
        logger.info(f"requested move: {move_index} for game {game_id}")
        return db_state

    def show_mode_on(self, game_id: str) -> None:
        if game_id in self._show_mode_jobs:
            raise ValueError(f"show mode already on for game_id {game_id}")
        db_state = self._games_db.get_state(game_id)
        agent = get_agent(db_state.size)
        db_node_store = DBNodeStore(game_id, self._games_db.db)
        self._show_mode_jobs[game_id] = ShowModeJob(game_id, db_state.boards[-1], agent, db_node_store, self._games_db)
        self._show_mode_jobs[game_id].start()

    def show_mode_off(self, game_id: str) -> None:
        if game_id not in self._show_mode_jobs:
            raise ValueError(f"show mode not on for game_id {game_id}")
        self._show_mode_jobs[game_id].stop()
        del self._show_mode_jobs[game_id]

    def list_show_mode_games(self) -> list[str]:
        return list(self._show_mode_jobs.keys())
