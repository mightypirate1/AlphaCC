import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.worker_stats import WorkerStats
from alpha_cc.engine import MCTS, Board


class MCTSAgent(Agent):
    def __init__(
        self,
        nn_service_addr: str,
        pred_channel: int = 0,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        rollout_gamma: float = 1.0,
        dirichlet_weight: float = 0.0,
        dirichlet_alpha: float = 0.15,
        c_puct_init: float = 2.0,
        c_puct_base: float = 10000.0,
        argmax_delay: int | None = None,
        n_threads: int = 1,
    ) -> None:
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._argmax_delay = argmax_delay
        self._steps_left_to_argmax = argmax_delay or np.inf
        self._mcts = MCTS(
            nn_service_addr,
            pred_channel,
            rollout_gamma,
            dirichlet_weight,
            dirichlet_alpha,
            c_puct_init,
            c_puct_base,
            n_threads,
        )

    def get_worker_stats(self) -> WorkerStats:
        return WorkerStats.from_fetch_stats(self._mcts.get_fetch_stats())

    @property
    def internal_nodes(self) -> dict[Board, MCTSNodePy]:
        return {board: MCTSNodePy.from_node(node) for board, node in self._mcts.get_nodes().items()}

    def on_game_start(self) -> None:
        self._steps_left_to_argmax = (self._argmax_delay or np.inf) + 1
        self._mcts.clear_nodes()

    def on_game_end(self) -> None:
        self._mcts.clear_nodes()

    def choose_move(self, board: Board, training: bool = False, temperature: float = 1.0) -> int:
        if self._argmax_delay is not None:
            self._steps_left_to_argmax -= 1
        pi, _ = self.run_rollouts(board, temperature=temperature)

        action_index = int(pi.argmax())
        if training and self._steps_left_to_argmax > 0:
            action_index = np.random.choice(len(pi), p=pi)
        return action_index

    def on_move_applied(self, action: int) -> None:
        """Notify the agent that a move was applied, so it can update internals (e.g. reroot the tree)."""
        self._mcts.advance_root(action)

    def run_rollouts(
        self,
        board: Board,
        temperature: float = 1.0,
        n_rollouts: int | None = None,
        rollout_depth: int | None = None,
    ) -> tuple[np.ndarray, float]:
        n_rollouts = n_rollouts if n_rollouts is not None else self._n_rollouts
        rollout_depth = rollout_depth if rollout_depth is not None else self._rollout_depth
        return self._mcts.run_rollouts(board, n_rollouts, rollout_depth, temperature)
