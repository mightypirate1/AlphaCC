from typing import List
import numpy as np
from alpha_cc import Board, BoardInfo
from agents.base_agent import BaseAgent, GameState

class GreedyAgent(BaseAgent):
    def __init__(self, size: int = 9):
        self.heuristic_matrix = np.ones((size, size))
        for x in range(self.heuristic_matrix.shape[0]):
            for y in range(self.heuristic_matrix.shape[1]):
                self.heuristic_matrix[x, y] = (x + y)

    def choose_move(self, board: Board) -> int:
        s_primes = self.unpack_s_primes(board)
        action = np.argmax(self._evaluation(s_primes))
        return action

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass

    def _evaluation(self, s_primes: List[GameState]) -> np.ndarray:
        def V(s):
            return ((np.array(s.matrix) == 1).astype(np.float) * self.heuristic_matrix).sum()
        return [V(sp) for sp in s_primes]
