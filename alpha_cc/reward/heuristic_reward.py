import numpy as np

from alpha_cc.agents.state.game_state import GameState
from alpha_cc.reward.reward_function import RewardFunction


class HeuristicReward(RewardFunction):
    def __init__(
        self,
        board_size: int,
        scale: float = 1.0,
        c_base: float = 1.0,
        c_tweak: float = 0.01,
        subtract_opponent: bool = False,
    ) -> None:
        def heuristic_reward(x: int, y: int) -> float:
            # basic rational:
            # - move forward down the middle
            # - if you are in the "home", move to the side to make space for others
            in_home = (x + y) >= 2 * board_size - ((board_size - 1) // 2 + 1)
            tweak_direction = -1 if in_home else 1
            linear_term = x + y
            cross_term = x * y
            return c_base * linear_term**0.9 + tweak_direction * c_tweak * cross_term**0.5

        self._subtract_opponent = subtract_opponent
        self._heuristic_matrix = np.ones((board_size, board_size))
        for x in range(self._heuristic_matrix.shape[0]):
            for y in range(self._heuristic_matrix.shape[1]):
                self._heuristic_matrix[x, y] = heuristic_reward(x, y)
        self._heuristic_matrix /= self._heuristic_matrix.sum()
        self._scale = scale

    def __call__(self, state: GameState) -> float:
        # TODO: figure out why this needs to be done the wrong way when everything else seems to be working
        other_player = 3 - state.info.current_player
        score = self._score(state.board.get_matrix_from_perspective_of_player(other_player))
        if self._subtract_opponent:
            flipped_board = state.board.get_matrix_from_perspective_of_current_player()
            opponent_score = self._score(flipped_board)
            score -= opponent_score
        return score

    def _score(self, matrix: list[list[int]]) -> float:
        # since current_player sees themselves as 1
        return float(self._scale * ((np.array(matrix) == 1) * self._heuristic_matrix).sum())
