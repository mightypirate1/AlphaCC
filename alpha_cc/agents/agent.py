from abc import ABC, abstractmethod

from alpha_cc.engine import Board


class Agent(ABC):
    @abstractmethod
    def choose_move(self, board: Board, training: bool = False) -> int:
        pass

    @abstractmethod
    def on_game_start(self) -> None:
        pass

    @abstractmethod
    def on_game_end(self) -> None:
        pass
