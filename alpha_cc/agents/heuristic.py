import numpy as np

from alpha_cc.engine import Board
from alpha_cc.state import GameState


class Heuristic:
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
            in_home = (GameState(board.reset()).matrix == 2)[x, y]
            tweak_direction = -1 if in_home else 1
            linear_term = x + y
            cross_term = x * y
            return c_base * linear_term**0.9 + tweak_direction * c_tweak * cross_term**0.5

        board = Board(board_size)
        self._subtract_opponent = subtract_opponent
        self._heuristic_matrix = np.ones((board_size, board_size))
        for x in range(self._heuristic_matrix.shape[0]):
            for y in range(self._heuristic_matrix.shape[1]):
                self._heuristic_matrix[x, y] = heuristic_reward(x, y)
        self._heuristic_matrix /= self._heuristic_matrix.sum()
        self._scale = scale

    def __call__(self, state: GameState, for_player: int | None = None) -> float:
        raw_matrix = state.board.get_matrix()
        matrix = np.array(raw_matrix)[: state.info.size, : state.info.size]
        player = state.info.current_player if for_player is None else for_player
        if player != state.info.current_player:
            matrix = 3 - matrix[::-1, ::-1]
        score = self._score(matrix)
        if self._subtract_opponent:
            score -= self._score(matrix[::-1, ::-1], player=3 - player)
        return score

    def _score(self, matrix: np.ndarray, player: int = 1) -> float:
        # since current_player sees themselves as 1
        return float(self._scale * ((np.array(matrix) == player) * self._heuristic_matrix).sum())
