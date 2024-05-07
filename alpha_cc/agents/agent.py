from abc import ABC, abstractmethod

import numpy as np

from alpha_cc.engine import Board


class Agent(ABC):
    @abstractmethod
    def choose_move(self, board: Board, training: bool = False) -> int | np.integer:
        pass

    @abstractmethod
    def on_game_start(self) -> None:
        pass

    @abstractmethod
    def on_game_end(self) -> None:
        pass
