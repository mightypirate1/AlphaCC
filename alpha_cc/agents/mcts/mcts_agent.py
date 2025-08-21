import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.engine import MCTS, Board


class MCTSAgent(Agent):
    def __init__(
        self,
        zmq_url: str,
        memcached_host: str,
        pred_channel: int = 0,
        cache_size: int = 300000,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        rollout_gamma: float = 1.0,
        dirichlet_weight: float = 0.0,
        dirichlet_alpha: float = 0.15,
        c_puct_init: float = 2.0,
        c_puct_base: float = 10000.0,
        argmax_delay: int | None = None,
    ) -> None:
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._argmax_delay = argmax_delay
        self._steps_left_to_argmax = argmax_delay or np.inf
        self._mcts = MCTS(
            zmq_url,
            memcached_host,
            pred_channel,
            cache_size,
            rollout_gamma,
            dirichlet_weight,
            dirichlet_alpha,
            c_puct_init,
            c_puct_base,
        )

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

    def run_rollouts(
        self,
        board: Board,
        temperature: float = 1.0,
        n_rollouts: int | None = None,
        rollout_depth: int | None = None,
    ) -> tuple[np.ndarray, float]:
        n_rollouts = n_rollouts if n_rollouts is not None else self._n_rollouts
        rollout_depth = rollout_depth if rollout_depth is not None else self._rollout_depth
        value = np.array([-self._mcts.run(board, rollout_depth) for _ in range(n_rollouts)]).mean()
        pi = self._rollout_policy(board, temperature)
        return pi, value

    def _rollout_policy(self, board: Board, temperature: float = 1.0) -> np.ndarray:
        node = self._mcts.get_node(board)

        # in case we did not do any rollouts yet, we default to uniform
        weighted_counts = np.ones(len(board.get_moves())) if node is None else np.array(node.n)
        if temperature != 1.0:  # save some flops
            weighted_counts = weighted_counts ** (1 / temperature)
        pi = weighted_counts / weighted_counts.sum()
        return pi
