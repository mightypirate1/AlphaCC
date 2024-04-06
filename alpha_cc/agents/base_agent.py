from abc import ABC, abstractmethod

from alpha_cc.agents.state import GameState
from alpha_cc.engine import Board


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
