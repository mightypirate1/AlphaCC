import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.worker_stats import WorkerStats
from alpha_cc.engine import MCTS, Board, RolloutResult


class MCTSAgent(Agent):
    def __init__(
        self,
        nn_service_addr: str,
        pred_channel: int = 0,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        rollout_gamma: float = 1.0,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        all_at_least_once: bool = False,
        base_count: int = 16,
        floor_count: int = 5,
        keep_frac: float = 0.5,
        argmax_delay: int | None = None,
        n_threads: int = 1,
        pruning_tree: bool = False,
        debug_prints: bool = False,
        dummy_preds: bool = False,
    ) -> None:
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._argmax_delay = argmax_delay
        self._steps_left_to_argmax = argmax_delay or np.inf
        self._mcts = MCTS(
            nn_service_addr,
            pred_channel,
            rollout_gamma,
            c_visit=c_visit,
            c_scale=c_scale,
            all_at_least_once=all_at_least_once,
            base_count=base_count,
            floor_count=floor_count,
            keep_frac=keep_frac,
            n_threads=n_threads,
            pruning_tree=pruning_tree,
            debug_prints=debug_prints,
            dummy_preds=dummy_preds,
        )

    def get_worker_stats(self) -> WorkerStats:
        return WorkerStats.from_fetch_stats(self._mcts.get_fetch_stats())

    @property
    def internal_nodes(self) -> dict[Board, MCTSNodePy]:
        return {board: MCTSNodePy.from_node(node) for board, node in self._mcts.get_nodes()}

    def on_game_start(self) -> None:
        self._steps_left_to_argmax = (self._argmax_delay or np.inf) + 1
        self._mcts.clear_nodes()

    def on_game_end(self) -> None:
        pass

    def on_move_applied(self, board: Board) -> None:
        self._mcts.on_move_applied(board)

    def choose_move(self, board: Board, training: bool = False, temperature: float = 1.0) -> int:
        if self._argmax_delay is not None:
            self._steps_left_to_argmax -= 1
        result = self.run_rollouts(board, temperature=temperature)

        action_index = int(result.pi.argmax())
        if training and self._steps_left_to_argmax > 0:
            action_index = np.random.choice(len(result.pi), p=result.pi)
        return action_index

    def run_rollouts(
        self,
        board: Board,
        temperature: float = 1.0,
        n_rollouts: int | None = None,
        rollout_depth: int | None = None,
    ) -> RolloutResult:
        n_rollouts = n_rollouts if n_rollouts is not None else self._n_rollouts
        rollout_depth = rollout_depth if rollout_depth is not None else self._rollout_depth
        return self._mcts.run_rollouts(board, n_rollouts, rollout_depth)
