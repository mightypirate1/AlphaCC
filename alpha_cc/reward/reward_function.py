from abc import ABC, abstractmethod

import numpy as np

from alpha_cc.agents.state.game_state import GameState


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, state: GameState) -> np.floating: ...
