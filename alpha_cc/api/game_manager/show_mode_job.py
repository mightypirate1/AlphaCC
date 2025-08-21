import logging
from datetime import datetime

from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler

from alpha_cc.agents.mcts.standalone_mcts_agent import StandaloneMCTSAgent
from alpha_cc.agents.mcts.node_store import DBNodeStore
from alpha_cc.db.games_db import GamesDB
from alpha_cc.db.models import DBGameState
from alpha_cc.engine import Board

scheduler = BackgroundScheduler()
scheduler.start()
logger = logging.getLogger(__name__)


class ShowModeJob:
    def __init__(
        self,
        game_id: str,
        board: Board,
        agent: StandaloneMCTSAgent,
        db_node_store: DBNodeStore,
        games_db: GamesDB,
        n_rollouts: int = 100,
        rollout_depth: int = 300,
    ) -> None:
        self._game_id = game_id
        self._agent = agent
        self._db_node_store = db_node_store
        self._games_db = games_db
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._job: Job | None = None
        self._games_db.set_show_mode_board(self._game_id, board)

    def start(self) -> None:
        def job() -> None:
            logger.info(f"show mode job started for game_id={self._game_id}")
            while (board := self._games_db.get_show_mode_board(self._game_id)) is not None:
                self._agent.run_rollouts(
                    board,
                    n_rollouts=self._n_rollouts,
                    rollout_depth=self._rollout_depth,
                )
                updated_nodes = self._agent.node_store.fetch_updated()
                self._db_node_store.load_from(updated_nodes)
                logger.info(f"show mode job updated nodes for game_id={self._game_id}")
            logger.info(f"show mode job terminated for game_id={self._game_id}")

        self._job = scheduler.add_job(job)
        self._job.modify(next_run_time=datetime.now())

    def stop(self) -> None:
        if self._job is not None:
            # this breaks the loop in the job
            self._games_db.clear_show_mode_board(self._game_id)

    def request_move(self) -> DBGameState:
        # make a move
        db_state = self._games_db.get_state(self._game_id)
        move_index = self._agent.choose_move(db_state.boards[-1])
        db_state.add_move(move_index)
        # update db
        self._games_db.add_move(self._game_id, move_index)
        self._games_db.set_show_mode_board(self._game_id, db_state.boards[-1])
        if db_state.boards[-1]:
            self.stop()
        return db_state

    def set_current_board(self, board: Board) -> None:
        self._games_db.set_show_mode_board(self._game_id, board)
