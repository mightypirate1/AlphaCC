from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from alpha_cc.agents.state import GameState

TTrainData = TypeVar("TTrainData")


class DualHeadNet(ABC, Generic[TTrainData]):
    @abstractmethod
    def policy(self, state: GameState) -> np.ndarray: ...

    @abstractmethod
    def value(self, state: GameState) -> np.floating: ...

    @abstractmethod
    def update_weights(self, train_data: TTrainData) -> None: ...