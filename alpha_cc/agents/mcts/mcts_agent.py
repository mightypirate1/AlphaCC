import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.engine import MCTS, Board


class MCTSAgent(Agent):
    def __init__(
        self,
        redis_host: str,
        cache_size: int = 1000000,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        dirichlet_weight: float = 0.0,
        dirichlet_alpha: float = 0.03,
        argmax_delay: int | None = None,
    ) -> None:
        self._redis_host = redis_host
        self._cache_size = cache_size
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._dirichlet_weight = dirichlet_weight
        self._dirichlet_alpha = dirichlet_alpha
        self._argmax_delay = argmax_delay
        self._steps_left_to_argmax = argmax_delay or np.inf
        self._mcts = MCTS(redis_host, cache_size, dirichlet_weight, dirichlet_alpha)

    def on_game_start(self) -> None:
        self._steps_left_to_argmax = (self._argmax_delay or np.inf) + 1
        self._mcts = MCTS(
            self._redis_host,
            self._cache_size,
            self._dirichlet_weight,
            self._dirichlet_alpha,
        )

    def on_game_end(self) -> None:
        pass

    def choose_move(self, board: Board, training: bool = False, temperature: float = 1.0) -> int | np.integer:
        if self._argmax_delay is not None:
            self._steps_left_to_argmax -= 1
        pi, _ = self.run_rollouts(board, temperature=temperature)
        if training and self._steps_left_to_argmax > 0:
            return np.random.choice(len(pi), p=pi)
        return pi.argmax()

    def run_rollouts(self, board: Board, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        value = np.array([-self._mcts.run(board, self._rollout_depth) for _ in range(self._n_rollouts)]).mean()
        pi = self._rollout_policy(board, temperature)
        return pi, value

    def _rollout_policy(self, board: Board, temperature: float = 1.0) -> np.ndarray:
        node = self._mcts.get_node(board)
        weighted_counts = np.array(node.n)
        if temperature != 1.0:  # save some flops
            weighted_counts = weighted_counts ** (1 / temperature)
        pi = weighted_counts / weighted_counts.sum()
        return pi
