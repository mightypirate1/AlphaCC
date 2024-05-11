from abc import ABC, abstractmethod

import numpy as np

from alpha_cc.state.game_state import GameState


class DualHeadEvaluator(ABC):
    @abstractmethod
    def policy(self, state: GameState) -> np.ndarray: ...

    @abstractmethod
    def value(self, state: GameState) -> np.floating | float: ...

    @abstractmethod
    def clear_cache(self) -> None: ...
