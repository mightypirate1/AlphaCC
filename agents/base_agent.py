from abc import ABC, abstractmethod
from typing import Callable, List
from alpha_cc import Board, BoardInfo


class GameState:
    info: BoardInfo
    matrix: List[List[int]]
    def __init__(self, board: Board):
        self.matrix = board.get_matrix_from_perspective_of_current_player()
        self.info = board.get_board_info()

class BaseAgent(ABC):

    def __init__(self):
        pass

    def unpack_state(self, board: Board) -> GameState:
        return GameState(board)

    def unpack_s_primes(self, board: Board) -> List[GameState]:
        return [GameState(b) for b in board.get_all_possible_next_states()]

    @abstractmethod
    def choose_action(self, board: Board) -> int:
        pass
