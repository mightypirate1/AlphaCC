import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.state import GameState
from alpha_cc.engine import Board
from alpha_cc.reward import HeuristicReward


class GreedyAgent(Agent):
    def __init__(self, size: int = 9) -> None:
        self._heuristic_function = HeuristicReward(size)

    def choose_move(self, board: Board, training: bool = False) -> int:  # noqa
        sp_values = self._evaluation(board)
        action = np.argmax(sp_values).astype(int)
        return action

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass

    def _evaluation(self, board: Board) -> np.ndarray:
        s_primes = [GameState(sp) for sp in board.get_all_possible_next_states()]
        return np.asarray([self._heuristic_function(sp) for sp in s_primes])
