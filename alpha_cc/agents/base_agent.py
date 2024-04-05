from abc import ABC, abstractmethod

from alpha_cc_engine import Board, BoardInfo


class GameState:
    info: BoardInfo
    matrix: list[list[int]]

    def __init__(self, board: Board) -> None:
        self.matrix = board.get_matrix_from_perspective_of_current_player()
        self.info = board.get_board_info()


class BaseAgent(ABC):
    def unpack_state(self, board: Board) -> GameState:
        return GameState(board)

    def unpack_s_primes(self, board: Board) -> list[GameState]:
        return [GameState(b) for b in board.get_all_possible_next_states()]

    @abstractmethod
    def choose_move(self, board: Board) -> int:
        pass

    @abstractmethod
    def on_game_start(self) -> None:
        pass

    @abstractmethod
    def on_game_end(self) -> None:
        pass
