import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.engine import Board
from alpha_cc.reward import HeuristicReward
from alpha_cc.state import GameState


class GreedyAgent(Agent):
    def __init__(self, size: int = 9) -> None:
        self._heuristic_function = HeuristicReward(size, subtract_opponent=True)

    def choose_move(self, board: Board, _: bool = False) -> int:
        sp_values = self._evaluation(board)
        action = int(np.argmax(sp_values))
        return action

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass

    def _evaluation(self, board: Board) -> np.ndarray:
        s_primes = GameState(board).children
        return np.asarray([self._heuristic_function(sp) for sp in s_primes])
