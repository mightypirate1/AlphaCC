import numpy as np

from alpha_cc.agents.base_agent import BaseAgent, GameState
from alpha_cc.engine import Board
from alpha_cc.reward import HeuristicReward


class GreedyAgent(BaseAgent):
    def __init__(self, size: int = 9) -> None:
        self._heuristic_function = HeuristicReward(size)

    def choose_move(self, board: Board) -> int:
        s_primes = self.unpack_s_primes(board)
        values = self._evaluation(s_primes)
        action = np.argmax(values).astype(int)
        return action

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass

    def _evaluation(self, s_primes: list[GameState]) -> np.ndarray:
        return np.asarray([self._heuristic_function(sp) for sp in s_primes])
