import numpy as np

from alpha_cc.agents.state.game_state import GameState
from alpha_cc.reward.reward_function import RewardFunction


class HeuristicReward(RewardFunction):
    def __init__(self, board_size: int, linear_weight: float = 1.0, cross_weight: float = 0.1) -> None:
        def heuristic_reward(x: int, y: int) -> float:
            return linear_weight * (x + y) + cross_weight * ((x * y) ** 0.5)

        self._heuristic_matrix = np.ones((board_size, board_size))
        for x in range(self._heuristic_matrix.shape[0]):
            for y in range(self._heuristic_matrix.shape[1]):
                self._heuristic_matrix[x, y] = heuristic_reward(x, y)
        self._heuristic_matrix /= self._heuristic_matrix.sum()

    def __call__(self, state: GameState) -> np.floating:
        # since current_player sees themselves as 1
        return ((np.array(state.matrix) == 1) * self._heuristic_matrix).sum()
